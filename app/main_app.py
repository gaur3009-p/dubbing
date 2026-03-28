"""
main_app.py  —  DubYou: Real-Time Identity-Preserving Voice Dubbing
====================================================================
Wires together every module in the repo into a single Gradio UI:

  Tabs
  ────
  1. 🎙 Streaming Dub   — chunk-by-chunk live dubbing with zero-shot voice cloning
  2. 📡 Live Mic        — rolling-buffer real-time with partial transcripts
  3. 🌍 Sentence / File — upload or record one sentence, full pipeline incl. TTS

  Pipeline per tab
  ────────────────
  Audio → VAD (Silero) → ASR (faster-whisper) → Translate (NLLB / Gemini)
       → TTS (CosyVoice zero-shot clone → Parler-TTS fallback) → output audio

Run
───
  python main_app.py
"""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gradio as gr
import numpy as np

# ── Pipelines ────────────────────────────────────────────────────────────────
from pipelines.streaming_pipeline import (
    run_pipeline   as run_stream,
    reset_stream,
    LANGUAGES,
)
from pipelines.live_pipeline import (
    run_live_pipeline as run_live,
    reset_live,
)
from pipelines.sentence_pipeline import run_travel_pipeline

# ── Voice identity (for enrollment status display) ────────────────────────────
from services.voice_identity import is_enrolled, clear_all as clear_voice_store

# ── Warm up all models at startup (prevents first-request delay) ──────────────
import services.asr
import services.asr_travel
import services.translator
import services.tts_engine
import services.tts_cloning
import realtime.vad
import realtime.chunker
import realtime.rolling_buffer

LANG_NAMES = sorted(LANGUAGES.keys())
_STREAM_SPEAKER = "stream_user"


# ═══════════════════════════════════════════════════════════════════════════════
#  Tab 1 — Streaming Dub  (latency-fixed + zero-shot voice cloning)
# ═══════════════════════════════════════════════════════════════════════════════

def _stream_cb(audio, target_lang):
    """
    Called every Gradio streaming frame.
    TTS now runs inside the thread pool (inside streaming_pipeline),
    so this callback never blocks on inference.
    """
    if audio is None:
        return "", "", None, "", _enroll_status()

    sr, data = audio
    data = _to_float32(data)
    transcript, translation, speech_file, latency = run_stream(data, sr, target_lang)
    return transcript, translation, speech_file, latency, _enroll_status()


def _stream_reset():
    reset_stream()
    clear_voice_store()
    return "", "", None, "", "⏳ Not enrolled — speak for ~3 s to activate cloning"


def _enroll_status() -> str:
    if is_enrolled(_STREAM_SPEAKER):
        return "✅ Voice cloned — dubbed audio matches your voice"
    return "⏳ Enrolling… speak for ~3 s to activate zero-shot cloning"


def build_streaming_tab():
    gr.Markdown(
        "### 🎙 Streaming Dub\n"
        "Speak into the mic. Audio is VAD-chunked → ASR → Translate → TTS, "
        "**all in a background thread pool** so the UI never freezes.\n\n"
        "🧬 **Zero-shot voice cloning** activates automatically after ~3 s of speech "
        "— the dubbed audio will sound like *your* voice."
    )
    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(
                sources=["microphone"],
                streaming=True,
                label="Your microphone",
            )
            lang_dd = gr.Dropdown(
                choices=LANG_NAMES,
                value="Hindi",
                label="Target language",
            )
            clone_status = gr.Textbox(
                label="🧬 Voice clone status",
                value="⏳ Not enrolled — speak for ~3 s to activate cloning",
                interactive=False,
                lines=1,
            )
            reset_btn = gr.Button("🔄 Reset session", variant="secondary")

        with gr.Column(scale=2):
            transcript_box  = gr.Textbox(label="📝 Transcript",  lines=6, interactive=False)
            translation_box = gr.Textbox(label="🌐 Translation", lines=6, interactive=False)
            audio_out       = gr.Audio(label="🔊 Dubbed audio",  autoplay=True)
            latency_box     = gr.Textbox(
                label="⏱ Latency  (ASR | Translate | TTS | Total | clone status)",
                interactive=False, lines=1,
            )

    audio_in.stream(
        fn=_stream_cb,
        inputs=[audio_in, lang_dd],
        outputs=[transcript_box, translation_box, audio_out, latency_box, clone_status],
    )

    reset_btn.click(
        fn=_stream_reset,
        outputs=[transcript_box, translation_box, audio_out, latency_box, clone_status],
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Tab 2 — Live Mic (rolling buffer + partial transcripts)
# ═══════════════════════════════════════════════════════════════════════════════

def _live_cb(audio, target_lang):
    if audio is None:
        transcript, translation, latency = run_live(None, 16000, target_lang)
    else:
        sr, data = audio
        data = _to_float32(data)
        transcript, translation, latency = run_live(data, sr, target_lang)
    return transcript, translation, latency


def _live_reset():
    reset_live()
    return "", "", ""


def build_live_tab():
    gr.Markdown(
        "### 📡 Live Mic\n"
        "Continuous rolling-buffer transcription with live **partial previews** "
        "(trailing `_` marks an in-progress word). "
        "Translation commits on silence (≥ 500 ms)."
    )
    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(
                sources=["microphone"],
                streaming=True,
                label="Microphone (live)",
            )
            lang_dd = gr.Dropdown(
                choices=LANG_NAMES,
                value="Hindi",
                label="Target language",
            )
            reset_btn = gr.Button("🔄 Reset session", variant="secondary")

        with gr.Column(scale=2):
            transcript_box  = gr.Textbox(label="📝 Transcript (live)",  lines=8, interactive=False)
            translation_box = gr.Textbox(label="🌐 Translation (live)", lines=8, interactive=False)
            latency_box     = gr.Textbox(label="⏱ Latency",             interactive=False, lines=1)

    audio_in.stream(
        fn=_live_cb,
        inputs=[audio_in, lang_dd],
        outputs=[transcript_box, translation_box, latency_box],
    )

    reset_btn.click(
        fn=_live_reset,
        outputs=[transcript_box, translation_box, latency_box],
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Tab 3 — Sentence / File  (full pipeline + voice cloning via reference upload)
# ═══════════════════════════════════════════════════════════════════════════════

def _sentence_cb(audio, ref_audio, target_lang):
    """
    Full pipeline for one recorded/uploaded sentence.
    If a reference voice clip is uploaded, zero-shot cloning is used for TTS.
    """
    if audio is None:
        return "", "", None, "No audio provided."

    sr, data = audio
    data = _to_float32(data)

    # ── Enroll from reference clip if provided ────────────────────────────────
    sentence_speaker = "sentence_user"
    if ref_audio is not None:
        ref_sr, ref_data = ref_audio
        ref_data = _to_float32(ref_data)
        from services.voice_identity import enroll_speaker, clear_speaker
        clear_speaker(sentence_speaker)
        enroll_speaker(sentence_speaker, ref_data, sr=ref_sr)

    # ── Run full pipeline inline ──────────────────────────────────────────────
    import time
    import tempfile
    import soundfile as sf
    import librosa
    from realtime.vad import detect_speech
    from services.asr_travel import transcribe_travel
    from services.translator import translate
    from services.tts_cloning import speak_as
    from services.voice_identity import is_enrolled
    from services.transcript_formatter import format_full_text

    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = np.clip(data.astype(np.float32), -1.0, 1.0)
    if sr != 16000:
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        sr = 16000

    if len(data) < 1600:
        return "", "", None, "Audio too short."

    segments = detect_speech(data, sr)
    if not segments:
        return "", "", None, "No speech detected."
    speech = data[segments[0]["start"]: segments[-1]["end"]]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, speech, sr)
        wav_path = tmp.name

    t0 = time.perf_counter()

    t_asr = time.perf_counter()
    raw_text, src_lang = transcribe_travel(wav_path)
    asr_ms = round((time.perf_counter() - t_asr) * 1000, 1)
    if not raw_text.strip():
        return "", "", None, "No transcript produced."

    text = format_full_text(raw_text) or raw_text.strip()

    tgt_code = LANGUAGES.get(target_lang, "hin_Deva")
    t_tr = time.perf_counter()
    translated = translate(text, src_lang, tgt_code)
    tr_ms = round((time.perf_counter() - t_tr) * 1000, 1)

    t_tts = time.perf_counter()
    speech_file = speak_as(translated.strip(), sentence_speaker)
    tts_ms = round((time.perf_counter() - t_tts) * 1000, 1)
    total_ms = round((time.perf_counter() - t0) * 1000, 1)

    clone_used = is_enrolled(sentence_speaker)
    latency = (
        f"ASR {asr_ms:.0f} ms  |  Translate {tr_ms:.0f} ms  |  "
        f"TTS {tts_ms:.0f} ms  |  Total {total_ms:.0f} ms  |  "
        f"voice clone: {'✅ used' if clone_used else '❌ no reference — generic voice'}"
    )
    return text, translated, speech_file, latency


def build_sentence_tab():
    gr.Markdown(
        "### 🌍 Sentence / File Dub\n"
        "Record or upload a sentence. Optionally upload a **reference voice clip** "
        "(3–10 s WAV/MP3 of the target speaker) to activate zero-shot voice cloning.\n\n"
        "Pipeline: VAD → Whisper-medium → NLLB → CosyVoice / Parler-TTS"
    )
    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(
                sources=["microphone", "upload"],
                type="numpy",
                label="🎤 Speech to dub",
            )
            ref_audio = gr.Audio(
                sources=["microphone", "upload"],
                type="numpy",
                label="🧬 Reference voice clip (optional — enables cloning)",
            )
            lang_dd = gr.Dropdown(
                choices=LANG_NAMES,
                value="Hindi",
                label="Target language",
            )
            run_btn = gr.Button("▶ Dub it", variant="primary")

        with gr.Column(scale=2):
            transcript_box  = gr.Textbox(label="📝 Transcript",  lines=5, interactive=False)
            translation_box = gr.Textbox(label="🌐 Translation", lines=5, interactive=False)
            audio_out       = gr.Audio(label="🔊 Dubbed audio",  autoplay=True)
            latency_box     = gr.Textbox(label="⏱ Latency",      interactive=False, lines=1)

    run_btn.click(
        fn=_sentence_cb,
        inputs=[audio_in, ref_audio, lang_dd],
        outputs=[transcript_box, translation_box, audio_out, latency_box],
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Shared helper
# ═══════════════════════════════════════════════════════════════════════════════

def _to_float32(data: np.ndarray) -> np.ndarray:
    """Normalise int16 or any integer PCM to float32 [-1, 1]."""
    if data.dtype == np.int16:
        return data.astype(np.float32) / 32768.0
    if np.issubdtype(data.dtype, np.integer):
        return data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    return data.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  Build & launch
# ═══════════════════════════════════════════════════════════════════════════════

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="DubYou — Real-Time Voice Dubbing",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown(
            "# 🎙 DubYou — Real-Time Identity-Preserving Voice Dubbing\n"
            "> *Same voice. Different language. Under 500 ms.*\n\n"
            "**Pipeline:** VAD (Silero) → ASR (Whisper) → Translate (NLLB / Gemini) "
            "→ TTS (CosyVoice zero-shot clone → Parler-TTS fallback)\n\n"
            "Set `DEEPGRAM_API_KEY` for faster ASR and `GEMINI_API_KEY` for faster translation."
        )

        with gr.Tabs():
            with gr.Tab("🎙 Streaming Dub"):
                build_streaming_tab()
            with gr.Tab("📡 Live Mic"):
                build_live_tab()
            with gr.Tab("🌍 Sentence / File"):
                build_sentence_tab()

        gr.Markdown(
            "---\n"
            "**Supported languages:** " + " · ".join(LANG_NAMES) + "\n\n"
            "**Voice cloning:** CosyVoice zero-shot (auto-installs if available) "
            "with automatic fallback to `ai4bharat/indic-parler-tts`.\n\n"
            "*All models load at startup — first run may take ~2 min while weights download.*"
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(
        share=True,
        server_port=7860,
        show_error=True,
    )
