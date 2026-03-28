"""
streaming_pipeline.py  —  Real-Time Dub Pipeline
=================================================

Three tiers, auto-selected by available API keys:

  Tier 1 — DEEPGRAM + GEMINI  (best, ~300ms total)
  ─────────────────────────────────────────────────
  Deepgram Nova-3 streams words back AS you speak (~120ms TTFT).
  Each final utterance fires Gemini Flash translation (~80ms).
  TTS runs immediately after translation in background thread.
  Semantic VAD (utterance_complete heuristic) decides when to commit.

  Tier 2 — DEEPGRAM only  (good, ~500ms)
  ───────────────────────────────────────
  Deepgram streaming ASR → NLLB translation (~300ms) → TTS.

  Tier 3 — Whisper + NLLB  (fallback, ~800ms, no keys needed)
  ─────────────────────────────────────────────────────────────
  Silero VAD chunks → faster-whisper → NLLB → TTS.
  Same reliable VAD-chunk approach that was working before.

All three tiers share:
  • Background thread pool for translate+TTS (Gradio never blocks)
  • results queue.Queue (exceptions never swallowed silently)
  • Plain list display (no TranscriptFormatter — was the bug source)
  • Voice clone toggle (speak_as vs speak_generic)
  • Auto voice enrollment from first 3s of speech
"""

import os, time, threading, queue, tempfile, asyncio
import numpy as np
import soundfile as sf
import librosa
import torch
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from silero_vad import load_silero_vad, get_speech_timestamps
from services.asr        import transcribe as whisper_transcribe
from services.translator import translate  as nllb_translate
from services.tts_engine import speak      as speak_generic
from services.tts_cloning    import speak_as
from services.voice_identity import enroll_speaker, is_enrolled
from realtime.semantic_vad   import is_utterance_complete

# ── API keys (set in Colab Cell 4) ───────────────────────────────────────────
DEEPGRAM_KEY = os.getenv("DEEPGRAM_API_KEY", "")
GEMINI_KEY   = os.getenv("GEMINI_API_KEY",   "")

# Lazy imports — only loaded when keys present
_gemini_translate = None
_deepgram_class   = None

def _load_fast_services():
    """Import fast services once, only if keys are available."""
    global _gemini_translate, _deepgram_class
    if GEMINI_KEY and _gemini_translate is None:
        try:
            from services.translator_gemini import translate_gemini_sync
            _gemini_translate = translate_gemini_sync
            print("[stream] Gemini translation: ✅ active")
        except Exception as e:
            print(f"[stream] Gemini unavailable: {e}")
    if DEEPGRAM_KEY and _deepgram_class is None:
        try:
            from services.asr_streaming import DeepgramStreamingASR
            _deepgram_class = DeepgramStreamingASR
            print("[stream] Deepgram ASR: ✅ active")
        except Exception as e:
            print(f"[stream] Deepgram unavailable: {e}")

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
SR             = 16000
MIN_SPEECH_MS  = 350
SILENCE_MS     = 450
MAX_CHUNK_MS   = 6000
ENROLL_SECS    = 3
SPEAKER_ID     = "stream_user"
MAX_LINES      = 8

# ── Silero VAD (Tier 3 fallback chunking) ────────────────────────────────────
_vad      = load_silero_vad()
_vad_lock = threading.Lock()

# ── Thread pool ───────────────────────────────────────────────────────────────
_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="dub")

# ── Results queue (thread → Gradio callback) ──────────────────────────────────
_results: queue.Queue = queue.Queue()

# ── Session state ─────────────────────────────────────────────────────────────
_buf:          np.ndarray  = np.array([], dtype=np.float32)
_speech_start: int | None  = None
_seq:          int         = 0
_use_cloning:  bool        = True
_target_lang:  str         = "Hindi"

_lines_t:   list[str]      = []
_lines_tr:  list[str]      = []
_pending:   dict[int, str] = {}   # seq → transcript (waiting for translate/TTS)

_enroll_buf:  list[np.ndarray] = []
_enroll_lock  = threading.Lock()

# Deepgram session (Tier 1/2) — one per streaming session
_dg_session   = None
_dg_thread    = None
_dg_loop      = None
_dg_interim:  str  = ""   # live partial from Deepgram
_dg_lock      = threading.Lock()


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


def _vad_ts(audio: np.ndarray):
    with _vad_lock:
        _vad.reset_states()
        return get_speech_timestamps(torch.from_numpy(audio), _vad, sampling_rate=SR)


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
            print(f"[stream] enrolled '{SPEAKER_ID}' ({total/SR:.1f}s)")


def _do_translate(text: str, src_lang: str, tgt_lang: str) -> tuple[str, float]:
    tgt_code = LANGUAGES.get(tgt_lang, "hin_Deva")
    t0 = time.perf_counter()
    if _gemini_translate:
        try:
            translated = _gemini_translate(text, src_lang, tgt_code)
            return translated, round((time.perf_counter() - t0) * 1000)
        except Exception as e:
            print(f"[stream] Gemini translate failed: {e}, falling back to NLLB")
    translated = nllb_translate(text, src_lang, tgt_code)
    return translated, round((time.perf_counter() - t0) * 1000)


def _build_display() -> tuple[str, str]:
    t_out  = _lines_t[:]
    tr_out = _lines_tr[:]
    for s in sorted(_pending.keys()):
        t_out.append(_pending[s])
        tr_out.append("⏳")
    return "\n".join(t_out[-MAX_LINES:]), "\n".join(tr_out[-MAX_LINES:])


def _tier_label() -> str:
    if _deepgram_class and _gemini_translate:
        return "🟢 Tier 1 — Deepgram + Gemini (~300ms)"
    elif _deepgram_class:
        return "🟡 Tier 2 — Deepgram + NLLB (~500ms)"
    return "🔴 Tier 3 — Whisper + NLLB (~800ms) — add API keys for faster"


# ── Background job: translate + TTS ──────────────────────────────────────────

def _job(transcript: str, src_lang: str, seq: int,
         tgt_lang: str, use_cloning: bool) -> None:
    try:
        translated, tr_ms = _do_translate(transcript, src_lang, tgt_lang)

        t_tts = time.perf_counter()
        if use_cloning and is_enrolled(SPEAKER_ID):
            speech_file = speak_as(translated.strip(), SPEAKER_ID)
        else:
            speech_file = speak_generic(translated.strip())
        tts_ms = round((time.perf_counter() - t_tts) * 1000)

        _results.put({
            "seq": seq, "transcript": transcript,
            "translation": translated.strip(),
            "speech_file": speech_file,
            "tr_ms": tr_ms, "tts_ms": tts_ms, "error": None,
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        _results.put({"seq": seq, "transcript": transcript,
                      "translation": "", "speech_file": None,
                      "tr_ms": 0, "tts_ms": 0, "error": str(e)})


def _submit(transcript: str, src_lang: str) -> None:
    global _seq
    if not transcript.strip():
        return
    _try_enroll(np.zeros(SR, dtype=np.float32))  # ensure enroll check runs
    _pending[_seq] = transcript.strip()
    _POOL.submit(_job, transcript.strip(), src_lang, _seq, _target_lang, _use_cloning)
    _seq += 1


# ── Tier 1/2: Deepgram streaming session ─────────────────────────────────────

def _start_deepgram_session() -> None:
    """Start a persistent Deepgram WebSocket in a background thread."""
    global _dg_session, _dg_thread, _dg_loop, _dg_interim

    if not _deepgram_class:
        return

    _dg_interim = ""

    def _run_loop():
        global _dg_loop, _dg_session
        loop = asyncio.new_event_loop()
        _dg_loop = loop

        async def _main():
            global _dg_session
            _dg_session = _deepgram_class(api_key=DEEPGRAM_KEY)
            await _dg_session.connect()

            async for text, is_final, _words in _dg_session.results():
                if text == "__utterance_end__":
                    continue
                with _dg_lock:
                    if is_final and is_utterance_complete(text, min_words=3):
                        _submit(text, "eng_Latn")
                        _dg_interim = ""
                    elif not is_final:
                        _dg_interim = text

        loop.run_until_complete(_main())

    _dg_thread = threading.Thread(target=_run_loop, daemon=True)
    _dg_thread.start()
    print("[stream] Deepgram session started")


def _stop_deepgram_session() -> None:
    global _dg_session, _dg_loop, _dg_interim
    _dg_interim = ""
    if _dg_session:
        try:
            _dg_session.close()
        except Exception:
            pass
        _dg_session = None


def _push_to_deepgram(frame: np.ndarray) -> None:
    if _dg_session and _dg_loop and _dg_loop.is_running():
        try:
            _dg_session.push(frame)
        except Exception:
            pass


# ── Tier 3: Silero VAD chunking ───────────────────────────────────────────────

def _vad_push(frame: np.ndarray) -> None:
    """Push frame into VAD buffer, submit chunk when speech ends."""
    global _buf, _speech_start

    _buf = np.concatenate([_buf, frame])
    max_buf = int((MAX_CHUNK_MS / 1000 + 1) * SR)
    if len(_buf) > max_buf:
        _buf = _buf[-max_buf:]

    if len(_buf) < int(0.1 * SR):
        return

    ts = _vad_ts(_buf)
    if not ts:
        _buf = np.array([], dtype=np.float32)
        _speech_start = None
        return

    first_start = ts[0]["start"]
    last_end    = ts[-1]["end"]
    silence_tail = len(_buf) - last_end

    if _speech_start is None:
        _speech_start = first_start

    speech_len     = last_end - _speech_start
    silence_needed = int(SILENCE_MS / 1000 * SR)
    min_speech     = int(MIN_SPEECH_MS / 1000 * SR)
    max_speech     = int(MAX_CHUNK_MS / 1000 * SR)

    if (silence_tail >= silence_needed and speech_len >= min_speech) \
            or speech_len >= max_speech:
        chunk = _buf[_speech_start: last_end].copy()
        _buf = _buf[last_end:]
        _speech_start = None
        _try_enroll(chunk)

        # Run Whisper in background
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, chunk, SR)
            wav_path = tmp.name

        def _whisper_then_submit(wp):
            try:
                text, src = whisper_transcribe(wp)
                if text.strip():
                    _submit(text, src or "eng_Latn")
            except Exception as e:
                print(f"[stream] Whisper error: {e}")

        _POOL.submit(_whisper_then_submit, wav_path)


# ── Public API ────────────────────────────────────────────────────────────────

def set_cloning(enabled: bool) -> None:
    global _use_cloning
    _use_cloning = enabled


def reset_stream() -> None:
    global _buf, _speech_start, _seq
    global _lines_t, _lines_tr, _pending, _enroll_buf

    _stop_deepgram_session()

    _buf          = np.array([], dtype=np.float32)
    _speech_start = None
    _seq          = 0
    _lines_t      = []
    _lines_tr     = []
    _pending      = {}

    with _enroll_lock:
        _enroll_buf.clear()
    while not _results.empty():
        try: _results.get_nowait()
        except: break

    # Re-start Deepgram if available
    if _deepgram_class:
        _start_deepgram_session()


def init_pipeline() -> None:
    """Call once at startup to load fast services and start Deepgram."""
    _load_fast_services()
    if _deepgram_class:
        _start_deepgram_session()


def run_pipeline(audio_frame: np.ndarray, sr: int, target_lang: str):
    """
    Called every Gradio streaming frame.
    Returns: transcript_str, translation_str, speech_file|None, latency_str
    """
    global _target_lang, _dg_interim

    _target_lang = target_lang

    # ── 1. Drain finished jobs ────────────────────────────────────────────────
    speech_file = None
    latency     = ""
    while not _results.empty():
        try:
            r = _results.get_nowait()
        except queue.Empty:
            break

        seq = r["seq"]
        _pending.pop(seq, None)

        if r.get("error") or not r.get("transcript"):
            continue

        _lines_t.append(r["transcript"])
        _lines_tr.append(r["translation"])
        if r.get("speech_file"):
            speech_file = r["speech_file"]

        tier = "Deepgram+Gemini" if (_deepgram_class and _gemini_translate) \
               else "Deepgram+NLLB" if _deepgram_class else "Whisper+NLLB"
        latency = (
            f"Translate {r['tr_ms']} ms | TTS {r['tts_ms']} ms | "
            f"{tier} | pending: {len(_pending)} | "
            f"clone: {'✅' if (is_enrolled(SPEAKER_ID) and _use_cloning) else '❌'}"
        )

    if audio_frame is None:
        t, tr = _build_display()
        return t, tr, speech_file, latency

    # ── 2. Normalise frame ────────────────────────────────────────────────────
    frame = _norm(audio_frame, sr)

    # ── 3. Route to correct tier ─────────────────────────────────────────────
    if _deepgram_class and _dg_session:
        # Tier 1 or 2: stream raw audio to Deepgram
        # Deepgram handles VAD + ASR; semantic_vad decides when to commit
        _push_to_deepgram(frame)
        _try_enroll(frame)
    else:
        # Tier 3: local Silero VAD + Whisper
        _vad_push(frame)

    # ── 4. Show Deepgram interim transcript live (Tier 1/2) ──────────────────
    t_display, tr_display = _build_display()

    with _dg_lock:
        interim = _dg_interim

    if interim:
        # Show interim in the last line of transcript with a cursor
        t_display  = (t_display + "\n" + interim + " ▌").lstrip("\n")
        tr_display = (tr_display + "\n" + "…").lstrip("\n")

    return t_display, tr_display, speech_file, latency
