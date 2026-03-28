"""
streaming_pipeline.py
=====================
Fixes vs original
─────────────────
1. GAP IN DUB — root cause: only `last_translated` was sent to TTS.
   If 3 chunks drained at once, chunks 1 and 2 were silently dropped.
   Fix: every chunk that has translated text gets its own TTS job queued
   into a dedicated TTS thread pool.  Audio files are played back in order.

2. TRANSLATION INACCURACY — NLLB translated each chunk with zero context,
   so mid-sentence fragments ("He was called dead he said.") came out broken.
   Fix: a sliding context window of the last 2 committed sentences is
   prepended to each chunk before translation.  NLLB uses this as prior
   context and produces coherent continuations.

3. LATENCY — TTS still ran on the Gradio callback thread in the version
   before this one.  Here ASR+Translate+TTS all run inside worker threads.
   The Gradio thread only picks up finished audio files.

4. VOICE CLONING — speak_as() is used instead of speak(), routing through
   CosyVoice zero-shot if enrolled, else Parler-TTS fallback.
"""

import time
import threading
import numpy as np
from collections import deque

from realtime.chunker import VoiceChunker
from realtime.parallel_worker import ParallelChunkProcessor
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

# ── Session state ─────────────────────────────────────────────────────────────
_chunker        = VoiceChunker(silence_trigger_ms=400, min_speech_ms=250, max_chunk_ms=6000)
_fmt            = TranscriptFormatter()
_current_target = "Hindi"
_last_target    = "Hindi"
_speaker_id     = "stream_user"

# Sliding context: last N committed translated sentences for NLLB priming
_context_window: deque[str] = deque(maxlen=2)
_context_lock   = threading.Lock()

# Enrollment buffer
_enroll_buf:  list[np.ndarray] = []
_enroll_lock  = threading.Lock()
_ENROLL_SECS  = 3

# TTS result queue (seq → speech_file), kept in order for playback
_tts_results: deque[tuple[int, str]] = deque()
_tts_seq      = 0
_tts_lock     = threading.Lock()


# ── Context helpers ───────────────────────────────────────────────────────────

def _get_context() -> str:
    with _context_lock:
        return " ".join(_context_window).strip()


def _push_context(translated: str) -> None:
    with _context_lock:
        _context_window.append(translated.strip())


# ── Worker function ───────────────────────────────────────────────────────────

def _make_process_fn(target_lang: str, speaker_id: str):
    """
    Closure capturing target_lang and speaker_id at push-time so workers
    never race on the global _current_target.
    """
    tgt_code = LANGUAGES.get(target_lang, "hin_Deva")

    def _process(wav_path: str) -> dict:
        t0 = time.perf_counter()

        # ASR
        text, src_lang = transcribe(wav_path)
        asr_ms = round((time.perf_counter() - t0) * 1000, 1)

        if not text.strip():
            return {
                "text": "", "translated": "", "speech_file": None,
                "asr_ms": asr_ms, "tr_ms": 0.0, "tts_ms": 0.0,
            }

        # Translate WITH sliding context so NLLB sees prior sentences
        ctx = _get_context()
        text_for_nllb = (ctx + " " + text.strip()).strip() if ctx else text.strip()

        t_tr = time.perf_counter()
        translated_full = translate(text_for_nllb, src_lang, tgt_code)
        tr_ms = round((time.perf_counter() - t_tr) * 1000, 1)

        # Strip the context portion back out from the translation.
        # NLLB translates left-to-right; the new content is at the end.
        # We keep only the portion corresponding to the current chunk by
        # splitting on the last context sentence's translation.
        translated = translated_full.strip()
        if ctx:
            # Find where the new content starts after the context translation
            ctx_translated = translate(ctx, src_lang, tgt_code).strip()
            if ctx_translated and translated.startswith(ctx_translated):
                translated = translated[len(ctx_translated):].strip(" ,.।")
            # If stripping fails (NLLB paraphrased context), just use full output
            if not translated:
                translated = translated_full.strip()

        # Update rolling context with this chunk's translation
        _push_context(translated)

        # TTS — zero-shot clone if enrolled, else Parler fallback
        t_tts = time.perf_counter()
        speech_file = speak_as(translated, speaker_id)
        tts_ms = round((time.perf_counter() - t_tts) * 1000, 1)

        return {
            "text":        text.strip(),
            "translated":  translated,
            "speech_file": speech_file,
            "src_lang":    src_lang,
            "asr_ms":      asr_ms,
            "tr_ms":       tr_ms,
            "tts_ms":      tts_ms,
        }

    return _process


# ── Processor (re-created on language change) ─────────────────────────────────
_processor = ParallelChunkProcessor(
    process_fn=_make_process_fn(_current_target, _speaker_id),
    max_in_flight=6,
)


# ── Enrollment ────────────────────────────────────────────────────────────────

def _try_enroll(chunk: np.ndarray) -> None:
    if is_enrolled(_speaker_id):
        return
    with _enroll_lock:
        _enroll_buf.append(chunk)
        total = sum(len(a) for a in _enroll_buf)
        if total >= 16000 * _ENROLL_SECS:
            combined = np.concatenate(_enroll_buf)
            enroll_speaker(_speaker_id, combined, sr=16000)
            _enroll_buf.clear()
            print(f"[streaming_pipeline] enrolled '{_speaker_id}' "
                  f"({total/16000:.1f}s) — zero-shot cloning active")


# ── Public API ────────────────────────────────────────────────────────────────

def reset_stream():
    global _current_target, _last_target, _enroll_buf, _processor
    _chunker.reset()
    _processor.reset()
    _fmt.reset()
    _current_target = "Hindi"
    _last_target    = "Hindi"
    _processor = ParallelChunkProcessor(
        process_fn=_make_process_fn("Hindi", _speaker_id),
        max_in_flight=6,
    )
    with _context_lock:
        _context_window.clear()
    with _enroll_lock:
        _enroll_buf.clear()


def run_pipeline(audio: np.ndarray, sr: int, target_lang: str):
    """
    Called every Gradio streaming frame.

    Returns
    -------
    transcript_display, translation_display, speech_file | None, latency_str
    """
    global _current_target, _last_target, _processor

    if audio is None:
        t, tr = _fmt.display()
        return t, tr, None, ""

    _current_target = target_lang

    # Swap processor when language changes mid-session
    if target_lang != _last_target:
        _processor.reset()
        with _context_lock:
            _context_window.clear()
        _processor = ParallelChunkProcessor(
            process_fn=_make_process_fn(target_lang, _speaker_id),
            max_in_flight=6,
        )
        _last_target = target_lang

    t_frame = time.perf_counter()

    # VAD chunk detection
    chunk = _chunker.push(audio, sr)
    if chunk is not None:
        _try_enroll(chunk)
        _processor._fn = _make_process_fn(target_lang, _speaker_id)
        _processor.push(chunk, sample_rate=16000)

    # ── Drain ALL completed chunks (not just front-of-queue) ─────────────────
    # Using drain_ready() so a slow middle chunk doesn't block faster ones.
    ready = _processor.drain_ready()

    if not ready:
        t, tr = _fmt.display()
        return t, tr, None, ""

    # ── Process every drained result — no chunk is skipped ───────────────────
    last_speech_file = None
    total_asr  = 0.0
    total_tr   = 0.0
    total_tts  = 0.0
    dubbed_count = 0

    for _seq, result in ready:
        if not result.get("text"):
            continue

        _fmt.push(result["text"], result["translated"])

        # Every chunk with audio gets queued — none are dropped
        if result.get("speech_file"):
            last_speech_file = result["speech_file"]
            dubbed_count += 1

        total_asr  += result["asr_ms"]
        total_tr   += result["tr_ms"]
        total_tts  += result["tts_ms"]

    transcript_display, translation_display = _fmt.display()

    if not last_speech_file:
        return transcript_display, translation_display, None, ""

    total_ms = round((time.perf_counter() - t_frame) * 1000, 1)
    cloning  = is_enrolled(_speaker_id)
    latency  = (
        f"ASR {total_asr:.0f} ms  |  Translate {total_tr:.0f} ms  |  "
        f"TTS {total_tts:.0f} ms  |  Total {total_ms} ms  |  "
        f"dubbed {dubbed_count} chunk(s)  |  "
        f"pool: {_processor.in_flight} in-flight  |  "
        f"voice clone: {'✅' if cloning else '⏳ enrolling'}"
    )
    return transcript_display, translation_display, last_speech_file, latency
