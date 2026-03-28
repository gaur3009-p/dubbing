"""
streaming_pipeline.py  —  VAD-gated Word-Accumulator Streaming Dub
===================================================================

Architecture
────────────
Every Gradio audio frame (~100 ms):

  frame → energy VAD gate
            │ silent → skip ASR entirely (no hallucinations)
            │ speech →
            │   rolling 2 s audio buffer → Whisper (small, beam=1)
            │   diff new words vs last transcript
            │   accumulate new words in word_buf
            │   COMMIT when:
            │     (≥3 words AND 250 ms silence) OR (≥8 words, no wait)
            │       → submit translate+TTS to thread pool (non-blocking)
            │       → store committed text separately (no "…" placeholder)
            │
  every frame → drain finished jobs
                  → update translation display
                  → return speech_file

Key fixes vs previous version
──────────────────────────────
1. HALLUCINATIONS — Whisper now runs ONLY when energy VAD says speech is
   present. Silent/noisy frames skip ASR entirely. This eliminates
   "EchoDestroy", "Venus", Chinese characters.

2. TRANSLATION SHOWING "…" — Removed the _fmt.push(commit_text, "…")
   placeholder entirely. Committed transcripts are stored in a separate
   _pending_lines list. TranscriptFormatter only receives finalised
   (text, translation) pairs when the job completes. The UI is built
   manually from committed lines + in-flight pending lines.

3. src_lang HARDCODED — Whisper's detected language is now passed
   correctly to NLLB. Works for any source language.

4. VOICE CLONE TOGGLE — use_cloning flag controls whether speak_as()
   (CosyVoice/Parler) or plain speak() (generic Parler) is called.
"""

import time
import threading
import numpy as np
import tempfile
import soundfile as sf
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future

from services.asr import transcribe
from services.translator import translate
from services.tts_cloning import speak_as
from services.tts_engine import speak as speak_generic
from services.voice_identity import enroll_speaker, is_enrolled
from services.transcript_formatter import _clean as clean_text

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

# ── Tuning constants ──────────────────────────────────────────────────────────
SR                  = 16000
WINDOW_SECS         = 2.0      # rolling Whisper window
COMMIT_WORDS        = 3        # min new words before commit check
FORCE_WORDS         = 8        # force commit regardless of silence
SILENCE_SEC         = 0.25     # silence needed to commit (seconds)
ENERGY_THRESHOLD    = 0.008    # RMS below this = silence
MIN_SPEECH_SEC      = 0.4      # ignore frames shorter than this
ENROLL_SECS         = 3        # seconds of audio needed for enrollment
SPEAKER_ID          = "stream_user"
MAX_DISPLAY_LINES   = 8

# ── Thread pool ───────────────────────────────────────────────────────────────
_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="dub_worker")

# ── Session state (reset by reset_stream) ─────────────────────────────────────
_audio_buf:       np.ndarray       = np.array([], dtype=np.float32)
_last_transcript: str              = ""
_word_buf:        list[str]        = []
_detected_lang:   str              = "eng_Latn"
_silence_frames:  int              = 0
_speech_frames:   int              = 0

# Display state — two parallel lists: committed lines
_committed_transcripts:  list[str] = []   # finalised transcript lines
_committed_translations: list[str] = []   # finalised translation lines
_pending_transcripts:    list[str] = []   # committed but translation not yet back

# Job queue
_job_queue: deque[tuple[int, Future]] = deque()
_job_seq:   int = 0
_queue_lock = threading.Lock()

# Context window for NLLB
_context_window: deque[str] = deque(maxlen=2)
_context_lock   = threading.Lock()

# Enrollment buffer
_enroll_buf:  list[np.ndarray] = []
_enroll_lock  = threading.Lock()

# Voice clone toggle
_use_cloning: bool = True


# ── Internal helpers ──────────────────────────────────────────────────────────

def _to_float32(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.int16:
        return data.astype(np.float32) / 32768.0
    if np.issubdtype(data.dtype, np.integer):
        return data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    return data.astype(np.float32)


def _resample(data: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == SR:
        return data
    import librosa
    return librosa.resample(data, orig_sr=src_sr, target_sr=SR)


def _rms(chunk: np.ndarray) -> float:
    return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))


def _get_context() -> str:
    with _context_lock:
        return " ".join(_context_window).strip()


def _push_context(text: str) -> None:
    with _context_lock:
        _context_window.append(text.strip())


def _try_enroll(frame: np.ndarray) -> None:
    if is_enrolled(SPEAKER_ID):
        return
    with _enroll_lock:
        _enroll_buf.append(frame.copy())
        total = sum(len(a) for a in _enroll_buf)
        if total >= SR * ENROLL_SECS:
            combined = np.concatenate(_enroll_buf)
            enroll_speaker(SPEAKER_ID, combined, sr=SR)
            _enroll_buf.clear()
            print(f"[stream] enrolled '{SPEAKER_ID}' ({total/SR:.1f}s)")


def _build_display() -> tuple[str, str]:
    """Build transcript and translation display strings from state."""
    t_lines  = _committed_transcripts[:]
    tr_lines = _committed_translations[:]

    # Add pending lines (translation not yet back — show "…" only here)
    for pt in _pending_transcripts:
        t_lines.append(pt)
        tr_lines.append("…")

    # Trim to max display lines
    t_lines  = t_lines[-MAX_DISPLAY_LINES:]
    tr_lines = tr_lines[-MAX_DISPLAY_LINES:]

    return "\n".join(t_lines), "\n".join(tr_lines)


# ── Background worker ─────────────────────────────────────────────────────────

def _translate_and_speak(
    text: str,
    src_lang: str,
    tgt_lang: str,
    speaker_id: str,
    use_cloning: bool,
) -> dict:
    tgt_code = LANGUAGES.get(tgt_lang, "hin_Deva")

    # NLLB with sliding context
    ctx = _get_context()
    text_with_ctx = (ctx + " " + text).strip() if ctx else text

    t_tr = time.perf_counter()
    translated_full = translate(text_with_ctx, src_lang, tgt_code)
    tr_ms = round((time.perf_counter() - t_tr) * 1000, 1)

    # Strip context from translation output
    translated = translated_full.strip()
    if ctx:
        try:
            ctx_tr = translate(ctx, src_lang, tgt_code).strip()
            if ctx_tr and translated.startswith(ctx_tr):
                translated = translated[len(ctx_tr):].lstrip(" ,।.")
        except Exception:
            pass
    if not translated:
        translated = translated_full.strip()

    _push_context(translated)

    # TTS
    t_tts = time.perf_counter()
    if use_cloning and is_enrolled(speaker_id):
        speech_file = speak_as(translated, speaker_id)
    else:
        speech_file = speak_generic(translated)
    tts_ms = round((time.perf_counter() - t_tts) * 1000, 1)

    return {
        "text":        text,
        "translated":  translated,
        "speech_file": speech_file,
        "tr_ms":       tr_ms,
        "tts_ms":      tts_ms,
    }


def _submit_job(text: str, src_lang: str, tgt_lang: str) -> None:
    global _job_seq
    fut = _POOL.submit(
        _translate_and_speak,
        text, src_lang, tgt_lang, SPEAKER_ID, _use_cloning
    )
    with _queue_lock:
        _job_queue.append((_job_seq, fut))
        _job_seq += 1


def _drain_jobs() -> list[dict]:
    done    = []
    pending = deque()
    with _queue_lock:
        while _job_queue:
            seq, fut = _job_queue.popleft()
            if fut.done():
                exc = fut.exception()
                if exc:
                    print(f"[stream] job {seq} error: {exc}")
                else:
                    done.append((seq, fut.result()))
            else:
                pending.append((seq, fut))
        for item in sorted(pending, key=lambda x: x[0]):
            _job_queue.append(item)
    done.sort(key=lambda x: x[0])
    return [r for _, r in done]


# ── Public API ────────────────────────────────────────────────────────────────

def set_cloning(enabled: bool) -> None:
    global _use_cloning
    _use_cloning = enabled


def reset_stream() -> None:
    global _audio_buf, _last_transcript, _word_buf, _detected_lang
    global _silence_frames, _speech_frames, _job_seq
    global _committed_transcripts, _committed_translations, _pending_transcripts

    _audio_buf              = np.array([], dtype=np.float32)
    _last_transcript        = ""
    _word_buf               = []
    _detected_lang          = "eng_Latn"
    _silence_frames         = 0
    _speech_frames          = 0
    _job_seq                = 0
    _committed_transcripts  = []
    _committed_translations = []
    _pending_transcripts    = []

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
    Called every Gradio streaming frame.
    Returns: transcript_str, translation_str, speech_file | None, latency_str
    """
    global _audio_buf, _last_transcript, _word_buf, _detected_lang
    global _silence_frames, _speech_frames
    global _committed_transcripts, _committed_translations, _pending_transcripts

    if audio_frame is None:
        t, tr = _build_display()
        return t, tr, None, ""

    # ── 1. Normalise frame ────────────────────────────────────────────────────
    frame = _to_float32(audio_frame)
    if frame.ndim > 1:
        frame = np.mean(frame, axis=1)
    frame = np.clip(_resample(frame, sr), -1.0, 1.0)

    # ── 2. Energy VAD gate ────────────────────────────────────────────────────
    rms = _rms(frame)
    is_speech = rms >= ENERGY_THRESHOLD

    if is_speech:
        _silence_frames  = 0
        _speech_frames  += 1
        _try_enroll(frame)
        # Accumulate into rolling buffer
        _audio_buf = np.concatenate([_audio_buf, frame])
        max_samples = int(WINDOW_SECS * SR)
        if len(_audio_buf) > max_samples:
            _audio_buf = _audio_buf[-max_samples:]
    else:
        _silence_frames += 1
        _speech_frames   = 0
        # Don't clear audio_buf — we need it for the final word diff after silence

    # ── 3. ASR — only when speech is present and buffer is long enough ────────
    asr_ms = 0.0
    silence_frames_needed = int(SILENCE_SEC / 0.10)   # ~2-3 frames

    run_asr = is_speech and len(_audio_buf) >= int(MIN_SPEECH_SEC * SR)

    if run_asr:
        t_asr = time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, _audio_buf, SR)
            wav_path = tmp.name
        new_transcript, detected = transcribe(wav_path)
        asr_ms = round((time.perf_counter() - t_asr) * 1000, 1)

        if detected:
            _detected_lang = detected

        # ── 4. Word diff ──────────────────────────────────────────────────────
        new_words  = new_transcript.strip().split()
        prev_words = _last_transcript.strip().split()

        if len(new_words) > len(prev_words):
            _word_buf.extend(new_words[len(prev_words):])

        _last_transcript = new_transcript

    # ── 5. Commit decision ────────────────────────────────────────────────────
    speech_file = None
    latency     = ""

    # Condition A: words buffered + speaker went quiet
    commit_on_silence = (
        len(_word_buf) >= COMMIT_WORDS
        and _silence_frames >= silence_frames_needed
    )
    # Condition B: too many words (fast talker)
    commit_on_overflow = len(_word_buf) >= FORCE_WORDS

    if (commit_on_silence or commit_on_overflow) and _word_buf:
        commit_text = " ".join(_word_buf).strip()
        _word_buf        = []
        _audio_buf       = np.array([], dtype=np.float32)
        _last_transcript = ""

        # Store in pending (shown as "…" in translation until job finishes)
        _pending_transcripts.append(clean_text(commit_text) or commit_text)

        # Fire translate+TTS in background
        _submit_job(commit_text, _detected_lang, target_lang)

    # ── 6. Drain finished jobs ────────────────────────────────────────────────
    finished = _drain_jobs()
    for result in finished:
        raw_text   = result["text"]
        translated = result["translated"]
        clean      = clean_text(raw_text) or raw_text

        # Remove from pending, add to committed
        if _pending_transcripts and _pending_transcripts[0] == clean:
            _pending_transcripts.pop(0)
        elif clean in _pending_transcripts:
            _pending_transcripts.remove(clean)

        _committed_transcripts.append(clean)
        _committed_translations.append(translated)

        if result.get("speech_file"):
            speech_file = result["speech_file"]

        latency = (
            f"ASR {asr_ms:.0f} ms  |  "
            f"Translate {result['tr_ms']:.0f} ms  |  "
            f"TTS {result['tts_ms']:.0f} ms  |  "
            f"queued: {len(_job_queue)}  |  "
            f"clone: {'✅' if (is_enrolled(SPEAKER_ID) and _use_cloning) else '❌'}"
        )

    transcript_display, translation_display = _build_display()
    return transcript_display, translation_display, speech_file, latency
