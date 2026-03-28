import time
import numpy as np

from realtime.rolling_buffer import RollingBuffer
from realtime.parallel_worker import ParallelChunkProcessor
from services.asr import transcribe
from services.asr_travel import transcribe_travel
from services.translator import translate
from services.transcript_formatter import TranscriptFormatter

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

_buffer         = RollingBuffer()
_fmt            = TranscriptFormatter()
_current_target = "Hindi"
_partial_text   = ""


def _run_partial(wav_path: str) -> dict:
    try:
        text, _ = transcribe(wav_path)
        return {"text": text.strip()}
    except Exception as e:
        print(f"[partial ASR error] {e}")
        return {"text": ""}


def _run_commit(wav_path: str) -> dict:
    try:
        t0 = time.perf_counter()
        text, src_lang = transcribe_travel(wav_path)
        asr_ms = round((time.perf_counter() - t0) * 1000, 1)

        if not text.strip():
            return {"text": "", "translated": "", "asr_ms": asr_ms, "tr_ms": 0.0}

        tgt_code   = LANGUAGES.get(_current_target, "hin_Deva")
        t_tr       = time.perf_counter()
        translated = translate(text.strip(), src_lang, tgt_code)
        tr_ms      = round((time.perf_counter() - t_tr) * 1000, 1)

        return {
            "text":       text.strip(),
            "translated": translated.strip(),
            "asr_ms":     asr_ms,
            "tr_ms":      tr_ms,
        }
    except Exception as e:
        print(f"[commit ASR error] {e}")
        return {"text": "", "translated": ""}


_partial_pool = ParallelChunkProcessor(process_fn=_run_partial, max_in_flight=2)
_commit_pool  = ParallelChunkProcessor(process_fn=_run_commit,  max_in_flight=8)

def reset_live():
    global _partial_text, _current_target
    _buffer.reset()
    _fmt.reset()
    _partial_pool.reset()
    _commit_pool.reset()
    _partial_text   = ""
    _current_target = "Hindi"


def run_live_pipeline(audio, sr: int, target_lang: str):
    """
    Returns (transcript_display, translation_display, latency_str).
    NEVER returns empty strings — always returns current display state.
    This prevents Gradio from blanking the boxes on None frames.
    """
    global _partial_text, _current_target

    _current_target = target_lang

    for _seq, result in _partial_pool.drain():
        if result.get("text"):
            _partial_text = result["text"]

    total_asr = 0.0
    total_tr  = 0.0
    committed = 0

    for _seq, result in _commit_pool.drain():
        if not result.get("text"):
            continue
        _fmt.push(result["text"], result.get("translated", ""))
        _partial_text = ""
        total_asr += result.get("asr_ms", 0)
        total_tr  += result.get("tr_ms", 0)
        committed += 1

    if audio is not None:
        arr = np.array(audio, dtype=np.float32) if not isinstance(audio, np.ndarray) else audio
        if arr.size > 0:
            partial_chunk, commit_chunk = _buffer.push(arr, sr)

            if partial_chunk is not None:
                _partial_pool.push(partial_chunk, sample_rate=16000)
            if commit_chunk is not None:
                _commit_pool.push(commit_chunk, sample_rate=16000)

    transcript_display, translation_display = _build_display()

    latency = ""
    if committed:
        latency = (
            f"ASR {total_asr:.0f} ms  |  Translate {total_tr:.0f} ms  |  "
            f"partial {_partial_pool.in_flight} / commit {_commit_pool.in_flight} in-flight"
        )

    return transcript_display, translation_display, latency


def _build_display():
    t_committed, tr_committed = _fmt.display()

    if _partial_text:
        transcript_display  = (t_committed + "\n" + _partial_text + "_").lstrip("\n")
        translation_display = (tr_committed + "\n" + "…").lstrip("\n")
    else:
        transcript_display  = t_committed
        translation_display = tr_committed

    return transcript_display, translation_display
