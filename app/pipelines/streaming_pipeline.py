import time
import threading
import numpy as np

from realtime.chunker import VoiceChunker
from realtime.parallel_worker import ParallelChunkProcessor
from services.asr import transcribe
from services.translator import translate
from services.tts_cloning import speak_as          # zero-shot → Parler fallback
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
_speaker_id     = "stream_user"          # single-user Gradio session

# Rolling enrollment buffer (accumulates raw 16 kHz float32 audio)
_enroll_buf: list[np.ndarray] = []
_enroll_lock = threading.Lock()
_ENROLL_SECONDS = 3                       # enroll after 3 s of speech


# ── Worker function (runs entirely in thread pool) ────────────────────────────

def _make_process_fn(target_lang: str, speaker_id: str):
    """
    Returns a closure that captures target_lang and speaker_id at push-time,
    so the worker never touches mutable globals.
    """
    tgt_code = LANGUAGES.get(target_lang, "hin_Deva")

    def _process(wav_path: str) -> dict:
        t0 = time.perf_counter()

        # ── ASR ───────────────────────────────────────────────────────────────
        text, src_lang = transcribe(wav_path)
        asr_ms = round((time.perf_counter() - t0) * 1000, 1)

        if not text.strip():
            return {"text": "", "translated": "", "speech_file": None,
                    "asr_ms": asr_ms, "tr_ms": 0.0, "tts_ms": 0.0}

        # ── Translate ─────────────────────────────────────────────────────────
        t_tr = time.perf_counter()
        translated = translate(text.strip(), src_lang, tgt_code)
        tr_ms = round((time.perf_counter() - t_tr) * 1000, 1)

        # ── TTS (zero-shot clone if enrolled, else Parler fallback) ───────────
        t_tts = time.perf_counter()
        speech_file = speak_as(translated.strip(), speaker_id)
        tts_ms = round((time.perf_counter() - t_tts) * 1000, 1)

        return {
            "text":        text.strip(),
            "translated":  translated.strip(),
            "speech_file": speech_file,
            "src_lang":    src_lang,
            "asr_ms":      asr_ms,
            "tr_ms":       tr_ms,
            "tts_ms":      tts_ms,
        }

    return _process


# ── Shared processor (process_fn is swapped per language change) ──────────────
_processor = ParallelChunkProcessor(
    process_fn=_make_process_fn("Hindi", _speaker_id),
    max_in_flight=4,            # reduced: TTS adds per-chunk cost
)
_last_target = "Hindi"


# ── Enrollment helper (called from main thread, thread-safe) ─────────────────

def _try_enroll(raw_chunk: np.ndarray) -> None:
    """Accumulate audio and enroll the speaker once enough data is gathered."""
    global _enroll_buf
    if is_enrolled(_speaker_id):
        return
    with _enroll_lock:
        _enroll_buf.append(raw_chunk)
        total = sum(len(a) for a in _enroll_buf)
        if total >= 16000 * _ENROLL_SECONDS:
            combined = np.concatenate(_enroll_buf)
            enroll_speaker(_speaker_id, combined, sr=16000)
            _enroll_buf.clear()
            print(f"[streaming_pipeline] speaker '{_speaker_id}' enrolled "
                  f"({total/16000:.1f} s) — zero-shot cloning active")


# ── Public API ────────────────────────────────────────────────────────────────

def reset_stream():
    global _current_target, _last_target, _enroll_buf
    _chunker.reset()
    _processor.reset()
    _fmt.reset()
    _current_target = "Hindi"
    _last_target    = "Hindi"
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

    # Swap process_fn when the user changes language mid-session
    if target_lang != _last_target:
        _processor.reset()
        _processor = ParallelChunkProcessor(
            process_fn=_make_process_fn(target_lang, _speaker_id),
            max_in_flight=4,
        )
        _last_target = target_lang

    t_frame = time.perf_counter()

    # Push raw audio into VAD chunker
    chunk = _chunker.push(audio, sr)
    if chunk is not None:
        _try_enroll(chunk)                      # enrollment from speech chunks
        fn = _make_process_fn(target_lang, _speaker_id)
        _processor._fn = fn                     # keep fn current
        _processor.push(chunk, sample_rate=16000)

    # Drain any completed chunks (non-blocking)
    ready = _processor.drain()
    if not ready:
        t, tr = _fmt.display()
        return t, tr, None, ""

    last_speech_file = None
    total_asr  = 0.0
    total_tr   = 0.0
    total_tts  = 0.0

    for _seq, result in ready:
        if not result.get("text"):
            continue
        _fmt.push(result["text"], result["translated"])
        if result.get("speech_file"):
            last_speech_file = result["speech_file"]
        total_asr  += result["asr_ms"]
        total_tr   += result["tr_ms"]
        total_tts  += result["tts_ms"]

    transcript_display, translation_display = _fmt.display()

    if not last_speech_file:
        return transcript_display, translation_display, None, ""

    total_ms = round((time.perf_counter() - t_frame) * 1000, 1)
    cloning_active = is_enrolled(_speaker_id)
    latency = (
        f"ASR {total_asr:.0f} ms  |  Translate {total_tr:.0f} ms  |  "
        f"TTS {total_tts:.0f} ms  |  Total {total_ms} ms  |  "
        f"pool: {_processor.in_flight} in-flight  |  "
        f"voice clone: {'✅ active' if cloning_active else '⏳ enrolling…'}"
    )
    return transcript_display, translation_display, last_speech_file, latency
