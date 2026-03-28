"""
main_app.py  —  DubYou: Real-Time Identity-Preserving Voice Dubbing
"""

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gradio as gr
import numpy as np

from pipelines.streaming_pipeline import (
    run_pipeline as run_stream,
    reset_stream,
    set_cloning,
    LANGUAGES,
)
from pipelines.live_pipeline      import run_live_pipeline as run_live, reset_live
from pipelines.sentence_pipeline  import run_travel_pipeline
from services.voice_identity      import is_enrolled, clear_all as clear_voice_store

import services.asr, services.asr_travel, services.translator
import services.tts_engine, services.tts_cloning
import realtime.vad, realtime.chunker, realtime.rolling_buffer

LANG_NAMES = sorted(LANGUAGES.keys())


def _to_float32(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.int16:
        return data.astype(np.float32) / 32768.0
    if np.issubdtype(data.dtype, np.integer):
        return data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    return data.astype(np.float32)


def _clone_status() -> str:
    if is_enrolled("stream_user"):
        return "✅ Voice enrolled — cloning ready"
    return "⏳ Not enrolled — speak ~3 s to auto-enroll"


# ═══════════════════════════════════════════════════════════════════════════════
#  Tab 1 — Streaming Dub
# ═══════════════════════════════════════════════════════════════════════════════

def _stream_cb(audio, target_lang, use_clone):
    if audio is None:
        return "", "", None, "", _clone_status()
    set_cloning(bool(use_clone))
    sr, data = audio
    transcript, translation, speech_file, latency = run_stream(
        _to_float32(data), sr, target_lang
    )
    return transcript, translation, speech_file, latency, _clone_status()


def _stream_reset(use_clone):
    reset_stream()
    clear_voice_store()
    set_cloning(bool(use_clone))
    return "", "", None, "", "⏳ Reset — speak ~3 s to enroll voice"


def build_streaming_tab():
    gr.Markdown(
        "### 🎙 Streaming Dub\n"
        "**Pipeline:** VAD gate → Whisper (rolling 2 s window) → "
        "word accumulator → translate + TTS in background thread.\n\n"
        "Commits after **3 new words + 250 ms silence**, or **8 words** regardless. "
        "No waiting for a full sentence pause."
    )
    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(
                sources=["microphone"], streaming=True, label="🎤 Microphone"
            )
            lang_dd = gr.Dropdown(
                choices=LANG_NAMES, value="Hindi", label="Target language"
            )

            # ── Voice clone toggle ────────────────────────────────────────────
            clone_toggle = gr.Checkbox(
                value=True,
                label="🧬 Use voice cloning (uncheck for generic TTS voice)",
                info="When checked, dubbed audio will sound like your voice after ~3 s enrollment.",
            )
            clone_status = gr.Textbox(
                label="Voice clone status",
                value="⏳ Not enrolled — speak ~3 s to auto-enroll",
                interactive=False,
                lines=1,
            )
            reset_btn = gr.Button("🔄 Reset session", variant="secondary")

        with gr.Column(scale=2):
            transcript_box = gr.Textbox(
                label="📝 Transcript", lines=6, interactive=False
            )
            translation_box = gr.Textbox(
                label="🌐 Translation", lines=6, interactive=False
            )
            audio_out = gr.Audio(label="🔊 Dubbed audio", autoplay=True)
            latency_box = gr.Textbox(
                label="⏱ Latency  (ASR | Translate | TTS | queued | clone)",
                interactive=False,
                lines=1,
            )

    audio_in.stream(
        fn=_stream_cb,
        inputs=[audio_in, lang_dd, clone_toggle],
        outputs=[transcript_box, translation_box, audio_out, latency_box, clone_status],
    )
    reset_btn.click(
        fn=_stream_reset,
        inputs=[clone_toggle],
        outputs=[transcript_box, translation_box, audio_out, latency_box, clone_status],
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Tab 2 — Live Mic
# ═══════════════════════════════════════════════════════════════════════════════

def _live_cb(audio, target_lang):
    if audio is None:
        return run_live(None, 16000, target_lang)
    sr, data = audio
    return run_live(_to_float32(data), sr, target_lang)


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
            reset_btn = gr.Button("🔄 Reset", variant="secondary")
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

def _sentence_cb(audio, ref_audio, target_lang, use_clone):
    if audio is None:
        return "", "", None, "No audio provided."

    sr, data = audio
    data = _to_float32(data)

    sentence_speaker = "sentence_user"
    if use_clone and ref_audio is not None:
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
    from services.tts_engine import speak as speak_gen
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

    segs = detect_speech(data, sr)
    if not segs:
        return "", "", None, "No speech detected."
    speech = data[segs[0]["start"]: segs[-1]["end"]]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, speech, sr)
        wav = tmp.name

    t0 = time.perf_counter()
    raw, src = transcribe_travel(wav)
    asr_ms = round((time.perf_counter() - t0) * 1000, 1)
    if not raw.strip():
        return "", "", None, "No transcript."
    text = format_full_text(raw) or raw.strip()

    tgt_code = LANGUAGES.get(target_lang, "hin_Deva")
    t_tr = time.perf_counter()
    translated = translate(text, src, tgt_code)
    tr_ms = round((time.perf_counter() - t_tr) * 1000, 1)

    t_tts = time.perf_counter()
    if use_clone and is_enrolled(sentence_speaker):
        sf_path = speak_as(translated.strip(), sentence_speaker)
        clone_note = "✅ cloned"
    else:
        sf_path = speak_gen(translated.strip())
        clone_note = "❌ generic (no ref clip)"
    tts_ms = round((time.perf_counter() - t_tts) * 1000, 1)
    total  = round((time.perf_counter() - t0) * 1000, 1)

    lat = (f"ASR {asr_ms:.0f} ms | Translate {tr_ms:.0f} ms | "
           f"TTS {tts_ms:.0f} ms | Total {total:.0f} ms | voice: {clone_note}")
    return text, translated, sf_path, lat


def build_sentence_tab():
    gr.Markdown(
        "### 🌍 Sentence / File Dub\n"
        "Record or upload one sentence. Provide a **reference voice clip** "
        "(3–10 s) to enable zero-shot voice cloning.\n\n"
        "Pipeline: VAD → Whisper-medium → NLLB → CosyVoice / Parler-TTS"
    )
    with gr.Row():
        with gr.Column(scale=1):
            audio_in    = gr.Audio(sources=["microphone","upload"], type="numpy",
                                   label="🎤 Speech to dub")
            ref_audio   = gr.Audio(sources=["microphone","upload"], type="numpy",
                                   label="🧬 Reference voice clip (optional)")
            lang_dd     = gr.Dropdown(choices=LANG_NAMES, value="Hindi",
                                      label="Target language")
            clone_toggle = gr.Checkbox(
                value=True,
                label="🧬 Use voice cloning",
                info="Requires a reference clip above. Falls back to generic TTS if not provided.",
            )
            run_btn = gr.Button("▶ Dub it", variant="primary")
        with gr.Column(scale=2):
            transcript_box  = gr.Textbox(label="📝 Transcript",  lines=5, interactive=False)
            translation_box = gr.Textbox(label="🌐 Translation", lines=5, interactive=False)
            audio_out       = gr.Audio(label="🔊 Dubbed audio",  autoplay=True)
            latency_box     = gr.Textbox(label="⏱ Latency",      interactive=False, lines=1)

    run_btn.click(
        fn=_sentence_cb,
        inputs=[audio_in, ref_audio, lang_dd, clone_toggle],
        outputs=[transcript_box, translation_box, audio_out, latency_box],
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Build & launch
# ═══════════════════════════════════════════════════════════════════════════════

def build_app() -> gr.Blocks:
    with gr.Blocks(title="DubYou", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🎙 DubYou — Real-Time Identity-Preserving Voice Dubbing\n"
            "> *Same voice. Different language. Under 500 ms.*\n\n"
            "**Pipeline:** Energy VAD → ASR (Whisper rolling window) → "
            "Translate (NLLB) → TTS (CosyVoice zero-shot / Parler-TTS)\n\n"
            "Set `DEEPGRAM_API_KEY` for faster ASR · `GEMINI_API_KEY` for faster translation."
        )
        with gr.Tabs():
            with gr.Tab("🎙 Streaming Dub"):   build_streaming_tab()
            with gr.Tab("📡 Live Mic"):         build_live_tab()
            with gr.Tab("🌍 Sentence / File"):  build_sentence_tab()

        gr.Markdown(
            "---\n**Languages:** " + " · ".join(LANG_NAMES) + "\n\n"
            "**Voice cloning:** auto-enrolls from mic · CosyVoice → Parler-TTS fallback.\n"
            "*Models load at startup — first run downloads weights (~2 min).*"
        )
    return demo


if __name__ == "__main__":
    build_app().launch(share=True, server_port=7860, show_error=True)
