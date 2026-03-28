"""
main_app.py  —  DubYou: Real-Time Identity-Preserving Voice Dubbing
====================================================================
Tabs
────
1. 🎙 Streaming Dub   — word-accumulator pipeline (new architecture)
2. 📡 Live Mic        — rolling-buffer with partial transcripts
3. 🌍 Sentence / File — full pipeline with optional voice reference

Run
───
  python main_app.py
"""

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gradio as gr
import numpy as np

from pipelines.streaming_pipeline import run_pipeline as run_stream, reset_stream, LANGUAGES
from pipelines.live_pipeline      import run_live_pipeline as run_live, reset_live
from pipelines.sentence_pipeline  import run_travel_pipeline
from services.voice_identity      import is_enrolled, clear_all as clear_voice_store

# warm up all models at startup
import services.asr, services.asr_travel, services.translator
import services.tts_engine, services.tts_cloning
import realtime.vad, realtime.chunker, realtime.rolling_buffer

LANG_NAMES = sorted(LANGUAGES.keys())


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_float32(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.int16:
        return data.astype(np.float32) / 32768.0
    if np.issubdtype(data.dtype, np.integer):
        return data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    return data.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  Tab 1 — Streaming Dub
# ═══════════════════════════════════════════════════════════════════════════════

def _stream_cb(audio, target_lang):
    if audio is None:
        return "", "", None, "", _clone_status()
    sr, data = audio
    data = _to_float32(data)
    transcript, translation, speech_file, latency = run_stream(data, sr, target_lang)
    return transcript, translation, speech_file, latency, _clone_status()


def _stream_reset():
    reset_stream()
    clear_voice_store()
    return "", "", None, "", "⏳ Not enrolled — will auto-enroll after ~3 s of speech"


def _clone_status() -> str:
    return ("✅ Voice cloned — dubbed audio matches your voice"
            if is_enrolled("stream_user")
            else "⏳ Enrolling… speak for ~3 s to activate zero-shot cloning")


def build_streaming_tab():
    gr.Markdown(
        "### 🎙 Streaming Dub\n"
        "**New architecture — word-accumulator pipeline:**\n"
        "Whisper runs on a rolling 2-second window every ~100 ms. "
        "As soon as 3 new words appear + 200 ms silence, translate+TTS fires "
        "immediately in the background — no waiting for a full sentence pause.\n\n"
        "🧬 Zero-shot voice cloning activates automatically after ~3 s of speech."
    )
    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(sources=["microphone"], streaming=True,
                                label="Your microphone")
            lang_dd  = gr.Dropdown(choices=LANG_NAMES, value="Hindi",
                                   label="Target language")
            clone_box = gr.Textbox(
                label="🧬 Voice clone status",
                value="⏳ Not enrolled — will auto-enroll after ~3 s of speech",
                interactive=False, lines=1)
            reset_btn = gr.Button("🔄 Reset session", variant="secondary")

        with gr.Column(scale=2):
            transcript_box  = gr.Textbox(label="📝 Transcript",  lines=6, interactive=False)
            translation_box = gr.Textbox(label="🌐 Translation", lines=6, interactive=False)
            audio_out       = gr.Audio(label="🔊 Dubbed audio",  autoplay=True)
            latency_box     = gr.Textbox(
                label="⏱ Latency  (ASR | Translate | TTS | jobs queued | clone)",
                interactive=False, lines=1)

    audio_in.stream(
        fn=_stream_cb,
        inputs=[audio_in, lang_dd],
        outputs=[transcript_box, translation_box, audio_out, latency_box, clone_box],
    )
    reset_btn.click(
        fn=_stream_reset,
        outputs=[transcript_box, translation_box, audio_out, latency_box, clone_box],
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Tab 2 — Live Mic
# ═══════════════════════════════════════════════════════════════════════════════

def _live_cb(audio, target_lang):
    if audio is None:
        t, tr, lat = run_live(None, 16000, target_lang)
    else:
        sr, data = audio
        t, tr, lat = run_live(_to_float32(data), sr, target_lang)
    return t, tr, lat


def build_live_tab():
    gr.Markdown(
        "### 📡 Live Mic\n"
        "Rolling-buffer transcription with live partial previews. "
        "Translation commits on ≥ 500 ms silence."
    )
    with gr.Row():
        with gr.Column(scale=1):
            audio_in  = gr.Audio(sources=["microphone"], streaming=True,
                                 label="Microphone (live)")
            lang_dd   = gr.Dropdown(choices=LANG_NAMES, value="Hindi",
                                    label="Target language")
            reset_btn = gr.Button("🔄 Reset session", variant="secondary")
        with gr.Column(scale=2):
            transcript_box  = gr.Textbox(label="📝 Transcript (live)",  lines=8, interactive=False)
            translation_box = gr.Textbox(label="🌐 Translation (live)", lines=8, interactive=False)
            latency_box     = gr.Textbox(label="⏱ Latency",             interactive=False, lines=1)

    audio_in.stream(fn=_live_cb, inputs=[audio_in, lang_dd],
                    outputs=[transcript_box, translation_box, latency_box])
    reset_btn.click(fn=lambda: ("", "", ""),
                    outputs=[transcript_box, translation_box, latency_box])


# ═══════════════════════════════════════════════════════════════════════════════
#  Tab 3 — Sentence / File
# ═══════════════════════════════════════════════════════════════════════════════

def _sentence_cb(audio, ref_audio, target_lang):
    if audio is None:
        return "", "", None, "No audio provided."

    sr, data = audio
    data = _to_float32(data)

    sentence_speaker = "sentence_user"
    if ref_audio is not None:
        ref_sr, ref_data = ref_audio
        ref_data = _to_float32(ref_data)
        from services.voice_identity import enroll_speaker, clear_speaker
        clear_speaker(sentence_speaker)
        enroll_speaker(sentence_speaker, ref_data, sr=ref_sr)

    import time, tempfile, librosa
    import soundfile as sf
    from realtime.vad import detect_speech
    from services.asr_travel import transcribe_travel
    from services.translator import translate
    from services.tts_cloning import speak_as
    from services.voice_identity import is_enrolled
    from services.transcript_formatter import format_full_text
    from pipelines.streaming_pipeline import LANGUAGES as LANG_MAP

    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = np.clip(data.astype(np.float32), -1.0, 1.0)
    if sr != 16000:
        data = librosa.resample(data, orig_sr=sr, target_sr=16000); sr = 16000
    if len(data) < 1600:
        return "", "", None, "Audio too short."

    segs = detect_speech(data, sr)
    if not segs:
        return "", "", None, "No speech detected."
    speech = data[segs[0]["start"]: segs[-1]["end"]]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, speech, sr); wav = tmp.name

    t0 = time.perf_counter()
    raw, src = transcribe_travel(wav); asr_ms = round((time.perf_counter()-t0)*1000,1)
    if not raw.strip(): return "", "", None, "No transcript."
    text = format_full_text(raw) or raw.strip()

    t_tr = time.perf_counter()
    translated = translate(text, src, LANG_MAP.get(target_lang, "hin_Deva"))
    tr_ms = round((time.perf_counter()-t_tr)*1000,1)

    t_tts = time.perf_counter()
    sf_path = speak_as(translated.strip(), sentence_speaker)
    tts_ms = round((time.perf_counter()-t_tts)*1000,1)
    total  = round((time.perf_counter()-t0)*1000,1)

    lat = (f"ASR {asr_ms:.0f} ms | Translate {tr_ms:.0f} ms | TTS {tts_ms:.0f} ms | "
           f"Total {total:.0f} ms | clone: {'✅' if is_enrolled(sentence_speaker) else '❌ generic'}")
    return text, translated, sf_path, lat


def build_sentence_tab():
    gr.Markdown(
        "### 🌍 Sentence / File Dub\n"
        "Upload or record one sentence. Optionally provide a **reference voice clip** "
        "(3–10 s) to enable zero-shot voice cloning.\n\n"
        "Pipeline: VAD → Whisper-medium → NLLB → CosyVoice / Parler-TTS"
    )
    with gr.Row():
        with gr.Column(scale=1):
            audio_in  = gr.Audio(sources=["microphone","upload"], type="numpy",
                                 label="🎤 Speech to dub")
            ref_audio = gr.Audio(sources=["microphone","upload"], type="numpy",
                                 label="🧬 Reference voice clip (optional)")
            lang_dd   = gr.Dropdown(choices=LANG_NAMES, value="Hindi",
                                    label="Target language")
            run_btn   = gr.Button("▶ Dub it", variant="primary")
        with gr.Column(scale=2):
            transcript_box  = gr.Textbox(label="📝 Transcript",  lines=5, interactive=False)
            translation_box = gr.Textbox(label="🌐 Translation", lines=5, interactive=False)
            audio_out       = gr.Audio(label="🔊 Dubbed audio",  autoplay=True)
            latency_box     = gr.Textbox(label="⏱ Latency",      interactive=False, lines=1)

    run_btn.click(fn=_sentence_cb,
                  inputs=[audio_in, ref_audio, lang_dd],
                  outputs=[transcript_box, translation_box, audio_out, latency_box])


# ═══════════════════════════════════════════════════════════════════════════════
#  Build & launch
# ═══════════════════════════════════════════════════════════════════════════════

def build_app() -> gr.Blocks:
    with gr.Blocks(title="DubYou — Real-Time Voice Dubbing", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🎙 DubYou — Real-Time Identity-Preserving Voice Dubbing\n"
            "> *Same voice. Different language. Under 500 ms.*\n\n"
            "**Pipeline:** VAD → ASR (Whisper rolling window) → "
            "Translate (NLLB / Gemini) → TTS (CosyVoice / Parler-TTS)\n\n"
            "Set `DEEPGRAM_API_KEY` for faster ASR and `GEMINI_API_KEY` for faster translation."
        )
        with gr.Tabs():
            with gr.Tab("🎙 Streaming Dub"):  build_streaming_tab()
            with gr.Tab("📡 Live Mic"):       build_live_tab()
            with gr.Tab("🌍 Sentence / File"): build_sentence_tab()

        gr.Markdown(
            "---\n**Languages:** " + " · ".join(LANG_NAMES) + "\n\n"
            "**Voice cloning:** CosyVoice zero-shot → Parler-TTS fallback.\n"
            "*Models load at startup — first run downloads weights (~2 min).*"
        )
    return demo


if __name__ == "__main__":
    build_app().launch(share=True, server_port=7860, show_error=True)
