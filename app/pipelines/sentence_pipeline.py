import tempfile
import soundfile as sf
import librosa
import numpy as np
import time
from realtime.vad import detect_speech
from services.asr_travel import transcribe_travel
from services.translator import translate
from services.tts_engine import speak
from services.transcript_formatter import format_full_text

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

def _preprocess(audio: np.ndarray, sr: int):
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    return audio, sr

def run_travel_pipeline(audio: np.ndarray, sr: int, target_lang: str):
    """
    Full pipeline for a recorded sentence.
    Returns (transcript_text, translated_text, speech_file, timings_dict).
    """
    t0 = time.perf_counter()
    audio, sr = _preprocess(audio, sr)
    if len(audio) < 1600:
        return "", "", None, {}
    segments = detect_speech(audio, sr)
    if not segments:
        return "", "", None, {}
    start = segments[0]["start"]
    end   = segments[-1]["end"]
    speech_audio = audio[start:end]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, speech_audio, sr)
        tmp_path = tmp.name
    t_asr = time.perf_counter()
    raw_text, src_lang = transcribe_travel(tmp_path)
    asr_ms = round((time.perf_counter() - t_asr) * 1000, 1)
    if not raw_text.strip():
        return "", "", None, {}
    text = format_full_text(raw_text)
    if not text.strip():
        text = raw_text.strip()
    t_tr = time.perf_counter()
    tgt_code   = LANGUAGES.get(target_lang, "hin_Deva")
    translated = translate(text, src_lang, tgt_code)
    tr_ms = round((time.perf_counter() - t_tr) * 1000, 1)
    t_tts = time.perf_counter()
    speech_file = speak(translated)
    tts_ms = round((time.perf_counter() - t_tts) * 1000, 1)
    total_ms = round((time.perf_counter() - t0) * 1000, 1)
    timings = {
        "asr":         asr_ms,
        "translation": tr_ms,
        "tts":         tts_ms,
        "total":       total_ms,
    }
    return text, translated, speech_file, timings
