"""
streaming_pipeline.py  —  Word-Accumulator Streaming Dub
=========================================================

Architecture (completely rewritten)
────────────────────────────────────

OLD approach (broken):
  mic → VAD chunk (500 ms silence) → ASR → translate → TTS
  Problem: nothing starts until the speaker pauses.
  One long sentence = one 2-3s stall. Parallel worker didn't help
  because ASR itself takes ~300ms and all steps were still serial
  per chunk.

NEW approach — Word-Accumulator Pipeline:
  ┌─────────────────────────────────────────────────────────────┐
  │  Gradio audio frame (every ~100ms)                          │
  │       ↓                                                     │
  │  Rolling 2-second audio window → Whisper (tiny, beam=1)    │
  │       ↓  incremental words appear                          │
  │  Word buffer — accumulate until COMMIT trigger:             │
  │    • ≥ 3 new words  AND  200ms silence  → COMMIT            │
  │    • ≥ 8 words (no silence yet)         → FORCE COMMIT      │
  │       ↓                                                     │
  │  Committed text → translate (NLLB, with context window)    │
  │       ↓                 ↑ fires in thread, non-blocking    │
  │  Translated text → TTS (Parler/CosyVoice)                  │
  │       ↓                 ↑ fires in thread, non-blocking    │
  │  speech_file → Gradio audio widget                         │
  └─────────────────────────────────────────────────────────────┘

Key design decisions
────────────────────
1. WHISPER runs on a 2s rolling window every frame (~100ms cadence).
   We diff the new transcript against the last one to extract *only
   new words* — no repeated translation of old words.

2. COMMIT is word-count + silence based, not VAD-chunk based.
   3 new confirmed words + 200ms quiet = commit immediately.
   This means "I don't know" fires translate+TTS in ~400ms from
   when the words were spoken, not after a 500ms silence gap.

3. TRANSLATE + TTS run in a single background thread per commit,
   submitted to a ThreadPoolExecutor. The Gradio thread never blocks
   on inference — it only queues jobs and drains finished ones.

4. ORDERED PLAYBACK: finished jobs are returned in submission order.
   A slow job does NOT block faster ones behind it (drain_ready logic).

5. CONTEXT WINDOW: last 2 committed translations are prepended to each
   new NLLB call so the model knows it's a continuation, not an
   isolated fragment.

6. VOICE CLONING: speak_as() is used — CosyVoice if enrolled,
   Parler-TTS fallback. Enrollment happens silently from the first
   3s of speech.
"""

import time
import threading
import numpy as np
import tempfile
import soundfile as sf
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future

# ── Services ──────────────────────────────────────────────────────────────────
from services.asr import transcribe
from services.translator import translate
from services.tts_cloning import speak_as
from services.voice_identity import enroll_speaker, is_enrolled
from services.transcript_formatter import TranscriptFormatter

# ── Language map ──────────────────────────────────────────────────────────────
LANGUAGES = {
    "English":   "eng_Latn",
    "Hindi":     "hin_Deva",
    "Bengali":   "ben_Beng",
    "Tamil":     "tam_Taml",
    "Telugu":    "tel_Telu",
    "Kannada":   "kan_Knda",
    "Malayalam": "mal_Mlym",
    "Marathi":   "mar_Deva",
    "Gujarati":  "guj_Gujr",
    "Punjabi":   "pan_Guru",
    "Urdu":      "urd_Arab",
    "Nepali":    "npi_Deva",
    "Odia":      "ory_Orya",
    "Assamese":  "asm_Beng",
    "Sindhi":    "snd_Arab",
    "Sanskrit":  "san_Deva",
}

# ── Constants ─────────────────────────────────────────────────────────────────
SR              = 16000
WINDOW_SECS     = 2.0          # rolling ASR window size
FRAME_HOP       = 0.10         # run ASR every 100ms
COMMIT_WORDS    = 3            # min new words to trigger commit
FORCE_WORDS     = 8            # force commit even without silence
SILENCE_COMMIT  = 0.20         # seconds of silence after ≥COMMIT_WORDS words
SILENCE_SAMPLES = int(SILENCE_COMMIT * SR)
ENROLL_SECS     = 3
SPEAKER_ID      = "stream_user"

# ── Thread pool (translate+TTS jobs) ─────────────────────────────────────────
_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="dub_worker")

# ── Session state ─────────────────────────────────────────────────────────────
_audio_buf:     np.ndarray       = np.array([], dtype=np.float32)
_last_transcript: str            = ""        # full transcript from last ASR run
_committed_words: int            = 0         # how many words we've already committed
_word_buf:      list[str]        = []        # new words not yet committed
_silence_count: int              = 0         # frames of silence seen
_fmt            = TranscriptFormatter()
_context_window: deque[str]      = deque(maxlen=2)
_context_lock   = threading.Lock()
_enroll_buf:    list[np.ndarray] = []
_enroll_lock    = threading.Lock()
_target_lang    = "Hindi"

# Job queue: (seq, Future[dict])
_job_queue: deque[tuple[int, Future]] = deque()
_job_seq:   int = 0
_queue_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_float32(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.int16:
        return data.astype(np.float32) / 32768.0
    if np.issubdtype(data.dtype, np.integer):
        return data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    return data.astype(np.float32)


def _resample(data: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return data
    import librosa
    return librosa.resample(data, orig_sr=sr, target_sr=SR)


def _is_silent(chunk: np.ndarray, threshold: float = 0.01) -> bool:
    """Energy-based silence check — fast, no VAD model needed."""
    return float(np.sqrt(np.mean(chunk ** 2))) < threshold


def _get_context() -> str:
    with _context_lock:
        return " ".join(_context_window).strip()


def _push_context(text: str) -> None:
    with _context_lock:
        _context_window.append(text.strip())


def _try_enroll(chunk: np.ndarray) -> None:
    if is_enrolled(SPEAKER_ID):
        return
    with _enroll_lock:
        _enroll_buf.append(chunk.copy())
        total = sum(len(a) for a in _enroll_buf)
        if total >= SR * ENROLL_SECS:
            combined = np.concatenate(_enroll_buf)
            enroll_speaker(SPEAKER_ID, combined, sr=SR)
            _enroll_buf.clear()
            print(f"[stream] enrolled '{SPEAKER_ID}' ({total/SR:.1f}s) — cloning active")


# ── Background worker: translate + TTS ───────────────────────────────────────

def _translate_and_speak(text: str, tgt_lang: str, speaker_id: str) -> dict:
    """Runs in thread pool. Returns speech_file path and timings."""
    tgt_code = LANGUAGES.get(tgt_lang, "hin_Deva")

    # Translate with sliding context so NLLB sees prior sentences
    ctx = _get_context()
    text_with_ctx = (ctx + " " + text).strip() if ctx else text

    t_tr = time.perf_counter()
    translated_full = translate(text_with_ctx, "eng_Latn", tgt_code)
    tr_ms = round((time.perf_counter() - t_tr) * 1000, 1)

    # Strip context portion back out
    translated = translated_full.strip()
    if ctx:
        ctx_tr = translate(ctx, "eng_Latn", tgt_code).strip()
        if ctx_tr and translated.startswith(ctx_tr):
            translated = translated[len(ctx_tr):].lstrip(" ,।.")
        if not translated:
            translated = translated_full.strip()

    _push_context(translated)

    # TTS
    t_tts = time.perf_counter()
    speech_file = speak_as(translated, speaker_id)
    tts_ms = round((time.perf_counter() - t_tts) * 1000, 1)

    return {
        "text":        text,
        "translated":  translated,
        "speech_file": speech_file,
        "tr_ms":       tr_ms,
        "tts_ms":      tts_ms,
    }


def _submit_job(text: str, tgt_lang: str) -> None:
    """Submit a translate+TTS job to the thread pool."""
    global _job_seq
    future = _POOL.submit(_translate_and_speak, text, tgt_lang, SPEAKER_ID)
    with _queue_lock:
        _job_queue.append((_job_seq, future))
        _job_seq += 1


def _drain_jobs() -> list[dict]:
    """
    Return all completed jobs in submission order, skipping still-running ones.
    A slow job does NOT block faster ones behind it.
    """
    done    = []
    pending = deque()

    with _queue_lock:
        while _job_queue:
            seq, fut = _job_queue.popleft()
            if fut.done():
                exc = fut.exception()
                if exc:
                    print(f"[stream] job {seq} failed: {exc}")
                else:
                    done.append((seq, fut.result()))
            else:
                pending.append((seq, fut))

        # Put unfinished jobs back, sorted by seq
        for item in sorted(pending, key=lambda x: x[0]):
            _job_queue.append(item)

    # Sort results by seq so playback order matches speech order
    done.sort(key=lambda x: x[0])
    return [r for _, r in done]


# ── Public API ────────────────────────────────────────────────────────────────

def reset_stream():
    global _audio_buf, _last_transcript, _committed_words, _word_buf
    global _silence_count, _target_lang, _job_seq, _enroll_buf
    _audio_buf       = np.array([], dtype=np.float32)
    _last_transcript = ""
    _committed_words = 0
    _word_buf        = []
    _silence_count   = 0
    _target_lang     = "Hindi"
    _job_seq         = 0
    _fmt.reset()
    with _context_lock:
        _context_window.clear()
    with _enroll_lock:
        _enroll_buf.clear()
    with _queue_lock:
        while _job_queue:
            _, fut = _job_queue.popleft()
            fut.cancel()


def run_pipeline(audio_frame: np.ndarray, sr: int, target_lang: str):
    """
    Called every Gradio streaming frame (~100ms cadence).

    Returns
    -------
    transcript_display, translation_display, speech_file | None, latency_str
    """
    global _audio_buf, _last_transcript, _committed_words
    global _word_buf, _silence_count, _target_lang

    if audio_frame is None:
        t, tr = _fmt.display()
        return t, tr, None, ""

    _target_lang = target_lang

    # ── 1. Ingest audio frame ─────────────────────────────────────────────────
    frame = _to_float32(audio_frame)
    if frame.ndim > 1:
        frame = np.mean(frame, axis=1)
    frame = np.clip(_resample(frame, sr), -1.0, 1.0)

    _audio_buf = np.concatenate([_audio_buf, frame])

    # Keep only the last WINDOW_SECS of audio
    max_samples = int(WINDOW_SECS * SR)
    if len(_audio_buf) > max_samples:
        _audio_buf = _audio_buf[-max_samples:]

    # ── 2. Enrollment from mic audio ─────────────────────────────────────────
    _try_enroll(frame)

    # ── 3. Silence detection (energy-based, very fast) ────────────────────────
    frame_is_silent = _is_silent(frame)
    if frame_is_silent:
        _silence_count += 1
    else:
        _silence_count = 0

    silence_frames_needed = int(SILENCE_COMMIT / FRAME_HOP)

    # ── 4. Run Whisper on rolling window (every frame) ────────────────────────
    t_asr = time.perf_counter()
    if len(_audio_buf) >= int(0.3 * SR):   # need at least 300ms for Whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, _audio_buf, SR)
            wav_path = tmp.name
        new_transcript, _ = transcribe(wav_path)
    else:
        new_transcript = _last_transcript
    asr_ms = round((time.perf_counter() - t_asr) * 1000, 1)

    # ── 5. Diff: find new words since last commit ─────────────────────────────
    new_words  = new_transcript.strip().split()
    prev_words = _last_transcript.strip().split()

    # New words = words beyond what we already had
    if len(new_words) > len(prev_words):
        incremental = new_words[len(prev_words):]
        _word_buf.extend(incremental)

    _last_transcript = new_transcript

    # ── 6. Commit decision ────────────────────────────────────────────────────
    should_commit = False

    # Condition A: enough new words AND speaker just went quiet
    if len(_word_buf) >= COMMIT_WORDS and _silence_count >= silence_frames_needed:
        should_commit = True

    # Condition B: too many words buffered (speaker talking fast, don't wait)
    if len(_word_buf) >= FORCE_WORDS:
        should_commit = True

    speech_file = None
    latency     = ""

    if should_commit and _word_buf:
        commit_text = " ".join(_word_buf).strip()
        _word_buf   = []

        # Reset rolling window so next ASR starts fresh after this commit
        _audio_buf       = np.array([], dtype=np.float32)
        _last_transcript = ""

        # Submit translate+TTS to background thread (non-blocking)
        _submit_job(commit_text, target_lang)
        _fmt.push(commit_text, "…")    # show transcript immediately

    # ── 7. Drain completed background jobs ───────────────────────────────────
    finished = _drain_jobs()
    for result in finished:
        _fmt.push(result["text"], result["translated"])
        if result.get("speech_file"):
            speech_file = result["speech_file"]
        latency = (
            f"ASR {asr_ms:.0f} ms  |  "
            f"Translate {result['tr_ms']:.0f} ms  |  "
            f"TTS {result['tts_ms']:.0f} ms  |  "
            f"jobs: {len(_job_queue)} queued  |  "
            f"clone: {'✅' if is_enrolled(SPEAKER_ID) else '⏳'}"
        )

    transcript_display, translation_display = _fmt.display()
    return transcript_display, translation_display, speech_file, latency
