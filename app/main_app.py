"""
main_app.py  —  DubYou: Real-Time Identity-Preserving Voice Dubbing
====================================================================
Wires together every module in the repo into a single Gradio UI:

  Tabs
  ────
  1. 🎙 Streaming Dub   — chunk-by-chunk live dubbing (VoiceChunker + ParallelChunkProcessor)
  2. 📡 Live Mic        — rolling-buffer real-time with partial transcripts
  3. 🌍 Sentence / File — upload or record one sentence, full pipeline incl. TTS

  Pipeline per tab
  ────────────────
  Audio → VAD (Silero) → ASR (faster-whisper) → Translate (NLLB / Gemini)
       → TTS (Parler / CosyVoice) → output audio + transcript

Run
───
  python main_app.py
"""

import sys
import os

# ── make sure the app/ directory is always importable ────────────────────────
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

# ── Services (imported so Gradio model-load happens at startup, not first use) ─
import services.asr          # loads faster-whisper small
import services.asr_travel   # loads faster-whisper medium
import services.translator   # loads NLLB-200
import services.tts_engine   # loads Parler-TTS
import realtime.vad          # loads Silero VAD
import realtime.chunker      # loads Silero VAD (chunker copy)
import realtime.rolling_buffer  # loads Silero VAD (rolling copy)

# ── Language list ─────────────────────────────────────────────────────────────
LANG_NAMES = sorted(LANGUAGES.keys())


# ═══════════════════════════════════════════════════════════════════════════════
#  Tab 1 — Streaming Dub
# ═══════════════════════════════════════════════════════════════════════════════

def _stream_cb(audio, target_lang, state):
    """Called every Gradio audio frame during streaming."""
    if audio is None:
        return state, "", None, ""
    sr, data = audio
    data = data.astype(np.float32) / (32768.0 if data.dtype == np.int16 else 1.0)
    transcript, translation, speech_file, latency = run_stream(data, sr, target_lang)
    return transcript, translation, speech_file, latency


def _stream_reset():
    reset_stream()
    return "", "", None, ""


def build_streaming_tab():
    gr.Markdown(
        "### 🎙 Streaming Dub\n"
        "Speak into the mic. Audio is VAD-chunked, transcribed, translated and dubbed "
        "in parallel. Latency breakdown is shown below the output."
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
            reset_btn = gr.Button("🔄 Reset session", variant="secondary")

        with gr.Column(scale=2):
            transcript_box  = gr.Textbox(label="📝 Transcript",   lines=6, interactive=False)
            translation_box = gr.Textbox(label="🌐 Translation",  lines=6, interactive=False)
            audio_out       = gr.Audio(label="🔊 Dubbed audio",   autoplay=True)
            latency_box     = gr.Textbox(label="⏱ Latency",       interactive=False, lines=1)

    # Wire streaming
    audio_in.stream(
        fn=_stream_cb,
        inputs=[audio_in, lang_dd, transcript_box],
        outputs=[transcript_box, translation_box, audio_out, latency_box],
    )

    reset_btn.click(
        fn=_stream_reset,
        outputs=[transcript_box, translation_box, audio_out, latency_box],
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Tab 2 — Live Mic (rolling buffer + partial transcripts)
# ═══════════════════════════════════════════════════════════════════════════════

def _live_cb(audio, target_lang):
    if audio is None:
        transcript, translation, latency = run_live(None, 16000, target_lang)
    else:
        sr, data = audio
        data = data.astype(np.float32) / (32768.0 if data.dtype == np.int16 else 1.0)
        transcript, translation, latency = run_live(data, sr, target_lang)
    return transcript, translation, latency


def _live_reset():
    reset_live()
    return "", "", ""


def build_live_tab():
    gr.Markdown(
        "### 📡 Live Mic\n"
        "Continuous rolling-buffer transcription with live partial previews. "
        "Translation is committed on utterance end (silence-triggered)."
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
            transcript_box  = gr.Textbox(label="📝 Transcript (live)",   lines=8, interactive=False)
            translation_box = gr.Textbox(label="🌐 Translation (live)",  lines=8, interactive=False)
            latency_box     = gr.Textbox(label="⏱ Latency",              interactive=False, lines=1)

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
#  Tab 3 — Sentence / File (full pipeline with TTS)
# ═══════════════════════════════════════════════════════════════════════════════

def _sentence_cb(audio, target_lang):
    """Process one recorded / uploaded sentence through the full pipeline."""
    if audio is None:
        return "", "", None, ""

    sr, data = audio
    data = data.astype(np.float32) / (32768.0 if data.dtype == np.int16 else 1.0)

    transcript, translation, speech_file, timings = run_travel_pipeline(
        data, sr, target_lang
    )

    if not timings:
        latency_str = "No speech detected."
    else:
        latency_str = (
            f"ASR {timings['asr']:.0f} ms  |  "
            f"Translate {timings['translation']:.0f} ms  |  "
            f"TTS {timings['tts']:.0f} ms  |  "
            f"Total {timings['total']:.0f} ms"
        )

    return transcript, translation, speech_file, latency_str


def build_sentence_tab():
    gr.Markdown(
        "### 🌍 Sentence / File Dub\n"
        "Record or upload a single sentence. Runs the **full pipeline**: "
        "VAD → Whisper medium → NLLB translation → Parler-TTS → dubbed audio."
    )
    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(
                sources=["microphone", "upload"],
                type="numpy",
                label="Record or upload audio",
            )
            lang_dd = gr.Dropdown(
                choices=LANG_NAMES,
                value="Hindi",
                label="Target language",
            )
            run_btn = gr.Button("▶ Dub it", variant="primary")

        with gr.Column(scale=2):
            transcript_box  = gr.Textbox(label="📝 Transcript",   lines=5, interactive=False)
            translation_box = gr.Textbox(label="🌐 Translation",  lines=5, interactive=False)
            audio_out       = gr.Audio(label="🔊 Dubbed audio",   autoplay=True)
            latency_box     = gr.Textbox(label="⏱ Latency",       interactive=False, lines=1)

    run_btn.click(
        fn=_sentence_cb,
        inputs=[audio_in, lang_dd],
        outputs=[transcript_box, translation_box, audio_out, latency_box],
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Build & launch the app
# ═══════════════════════════════════════════════════════════════════════════════

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="DubYou — Real-Time Voice Dubbing",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown(
            "# 🎙 DubYou — Real-Time Identity-Preserving Voice Dubbing\n"
            "> *Same voice. Different language. Under 500 ms.*\n\n"
            "**Pipeline:** VAD (Silero) → ASR (Whisper) → Translate (NLLB / Gemini) → TTS (Parler-TTS)\n\n"
            "Set `DEEPGRAM_API_KEY` for faster ASR (~120 ms) and `GEMINI_API_KEY` for faster translation (~80 ms)."
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
            "**Supported languages:** "
            + " · ".join(LANG_NAMES)
            + "\n\n"
            "*Models loaded at startup — first run may be slow while weights download.*"
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(
        share=True,        # creates a public URL (needed in Colab)
        server_port=7860,
        show_error=True,
    )
