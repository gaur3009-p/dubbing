"""
streaming_pipeline.py
=====================
Architecture: VAD-chunk → ASR → Translate → TTS, all in background threads.
The Gradio callback never blocks on inference.

Why previous approaches failed
───────────────────────────────
1. Word-diff on rolling Whisper window: Whisper re-transcribes differently
   each frame, diffs produce garbage, hallucinations pile up.
2. _fmt.push(text, "…") placeholder: TranscriptFormatter appends rather
   than replaces, so "…" accumulated forever.
3. Background thread exceptions were swallowed silently — translation
   never completed so translation box stayed "…".

This version
────────────
• Silero VAD chunks audio. When a chunk is ready it goes to a thread.
• Thread does: ASR → translate → TTS. All three steps together.
• Results queue: main thread drains it every frame (non-blocking).
• Display: two plain lists (transcripts[], translations[]).
  Pending lines shown with "⏳" while job runs. Replaced when done.
• No TranscriptFormatter (it was the bug source for streaming).
• Exceptions are caught per-job and printed — never silently dropped.
• Voice clone toggle respected.
"""

import time
import threading
import queue
import tempfile
import numpy as np
import soundfile as sf
import torch
import librosa
from concurrent.futures import ThreadPoolExecutor

from silero_vad import load_silero_vad, get_speech_timestamps
from services.asr       import transcribe
from services.translator import translate
from services.tts_engine import speak as speak_generic
from services.tts_cloning import speak_as
from services.voice_identity import enroll_speaker, is_enrolled

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

# ── VAD model ─────────────────────────────────────────────────────────────────
_vad = load_silero_vad()
_vad_lock = threading.Lock()

SR                = 16000
MIN_SPEECH_MS     = 400       # ignore VAD chunks shorter than this
SILENCE_MS        = 500       # silence needed to flush a chunk
MAX_CHUNK_MS      = 7000      # hard max chunk length
ENROLL_SECS       = 3
SPEAKER_ID        = "stream_user"
MAX_LINES         = 8

# ── Thread pool ───────────────────────────────────────────────────────────────
_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="dub")

# ── Results queue (thread → main) ─────────────────────────────────────────────
# Each item: {"seq": int, "transcript": str, "translation": str,
#             "speech_file": str|None, "tr_ms": float, "tts_ms": float, "error": str|None}
_results: queue.Queue = queue.Queue()

# ── Session state ─────────────────────────────────────────────────────────────
_buf:            np.ndarray  = np.array([], dtype=np.float32)
_speech_start:   int | None  = None
_seq_counter:    int         = 0
_use_cloning:    bool        = True
_target_lang:    str         = "Hindi"

# Display lists
_lines_t:   list[str] = []   # committed transcript lines
_lines_tr:  list[str] = []   # committed translation lines
_pending:   dict[int, str] = {}   # seq → transcript text (pending translation)

# Enrollment buffer
_enroll_buf:  list[np.ndarray] = []
_enroll_lock  = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm(data: np.ndarray, sr: int) -> np.ndarray:
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    else:
        data = data.astype(np.float32)
    data = np.clip(data, -1.0, 1.0)
    if sr != SR:
        data = librosa.resample(data, orig_sr=sr, target_sr=SR)
    return data


def _vad_timestamps(audio: np.ndarray):
    with _vad_lock:
        _vad.reset_states()
        return get_speech_timestamps(
            torch.from_numpy(audio), _vad, sampling_rate=SR
        )


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


def _build_display() -> tuple[str, str]:
    t_out  = _lines_t[:]
    tr_out = _lines_tr[:]
    # add pending lines (in seq order)
    for seq in sorted(_pending.keys()):
        t_out.append(_pending[seq])
        tr_out.append("⏳ translating…")
    t_out  = t_out[-MAX_LINES:]
    tr_out = tr_out[-MAX_LINES:]
    return "\n".join(t_out), "\n".join(tr_out)


# ── Background job ────────────────────────────────────────────────────────────

def _job(wav_path: str, seq: int, tgt_lang: str, use_cloning: bool) -> None:
    """Runs in thread pool. Puts result into _results queue."""
    try:
        # ASR
        t0 = time.perf_counter()
        transcript, src_lang = transcribe(wav_path)
        asr_ms = round((time.perf_counter() - t0) * 1000)

        if not transcript.strip():
            _results.put({"seq": seq, "transcript": "", "translation": "",
                          "speech_file": None, "asr_ms": asr_ms,
                          "tr_ms": 0, "tts_ms": 0, "error": None})
            return

        # Translate
        tgt_code = LANGUAGES.get(tgt_lang, "hin_Deva")
        t_tr = time.perf_counter()
        translation = translate(transcript.strip(), src_lang or "eng_Latn", tgt_code)
        tr_ms = round((time.perf_counter() - t_tr) * 1000)

        # TTS
        t_tts = time.perf_counter()
        if use_cloning and is_enrolled(SPEAKER_ID):
            speech_file = speak_as(translation.strip(), SPEAKER_ID)
        else:
            speech_file = speak_generic(translation.strip())
        tts_ms = round((time.perf_counter() - t_tts) * 1000)

        _results.put({
            "seq":         seq,
            "transcript":  transcript.strip(),
            "translation": translation.strip(),
            "speech_file": speech_file,
            "asr_ms":      asr_ms,
            "tr_ms":       tr_ms,
            "tts_ms":      tts_ms,
            "error":       None,
        })

    except Exception as e:
        print(f"[stream job {seq}] EXCEPTION: {e}")
        import traceback; traceback.print_exc()
        _results.put({"seq": seq, "transcript": "", "translation": "",
                      "speech_file": None, "asr_ms": 0,
                      "tr_ms": 0, "tts_ms": 0, "error": str(e)})


def _submit(chunk: np.ndarray) -> None:
    global _seq_counter
    _try_enroll(chunk)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, chunk, SR)
        wav_path = tmp.name
    seq = _seq_counter
    _seq_counter += 1
    # add to pending immediately so display shows ⏳
    _pending[seq] = "…"
    _POOL.submit(_job, wav_path, seq, _target_lang, _use_cloning)


# ── Public API ────────────────────────────────────────────────────────────────

def set_cloning(enabled: bool) -> None:
    global _use_cloning
    _use_cloning = enabled


def reset_stream() -> None:
    global _buf, _speech_start, _seq_counter
    global _lines_t, _lines_tr, _pending, _enroll_buf
    _buf          = np.array([], dtype=np.float32)
    _speech_start = None
    _seq_counter  = 0
    _lines_t      = []
    _lines_tr     = []
    _pending      = {}
    with _enroll_lock:
        _enroll_buf.clear()
    # drain stale results
    while not _results.empty():
        try: _results.get_nowait()
        except: break


def run_pipeline(audio_frame: np.ndarray, sr: int, target_lang: str):
    """
    Called every Gradio streaming frame (~100–200 ms).
    Returns: transcript_str, translation_str, speech_file|None, latency_str
    """
    global _buf, _speech_start, _target_lang

    _target_lang = target_lang

    # ── 1. Drain finished jobs first (always, even on None frame) ────────────
    speech_file = None
    latency     = ""
    while not _results.empty():
        try:
            r = _results.get_nowait()
        except queue.Empty:
            break

        seq = r["seq"]
        _pending.pop(seq, None)

        if r.get("error"):
            continue
        if not r["transcript"]:
            continue

        _lines_t.append(r["transcript"])
        _lines_tr.append(r["translation"])
        if r.get("speech_file"):
            speech_file = r["speech_file"]
        latency = (
            f"ASR {r['asr_ms']} ms | "
            f"Translate {r['tr_ms']} ms | "
            f"TTS {r['tts_ms']} ms | "
            f"pending: {len(_pending)} | "
            f"clone: {'✅' if (is_enrolled(SPEAKER_ID) and _use_cloning) else '❌'}"
        )

    if audio_frame is None:
        t, tr = _build_display()
        return t, tr, speech_file, latency

    # ── 2. Normalise incoming frame ───────────────────────────────────────────
    frame = _norm(audio_frame, sr)
    _buf  = np.concatenate([_buf, frame])

    # Cap buffer to avoid unbounded growth
    max_buf = int((MAX_CHUNK_MS / 1000 + 1) * SR)
    if len(_buf) > max_buf:
        _buf = _buf[-max_buf:]

    # ── 3. VAD: find speech segments in buffer ────────────────────────────────
    if len(_buf) < int(0.1 * SR):
        t, tr = _build_display()
        return t, tr, speech_file, latency

    timestamps = _vad_timestamps(_buf)

    if not timestamps:
        # Silence — if we had speech before, it's now gone → flush nothing
        # (buffer trimmed to avoid accumulating noise)
        _buf          = np.array([], dtype=np.float32)
        _speech_start = None
        t, tr = _build_display()
        return t, tr, speech_file, latency

    first_start  = timestamps[0]["start"]
    last_end     = timestamps[-1]["end"]
    silence_tail = len(_buf) - last_end       # samples of silence after speech

    if _speech_start is None:
        _speech_start = first_start

    speech_len     = last_end - _speech_start
    silence_needed = int(SILENCE_MS / 1000 * SR)
    min_speech     = int(MIN_SPEECH_MS / 1000 * SR)
    max_speech     = int(MAX_CHUNK_MS / 1000 * SR)

    flush = (
        (silence_tail >= silence_needed and speech_len >= min_speech)
        or speech_len >= max_speech
    )

    if flush:
        chunk = _buf[_speech_start: last_end].copy()
        _buf          = _buf[last_end:]
        _speech_start = None
        _submit(chunk)

    t, tr = _build_display()
    return t, tr, speech_file, latency
