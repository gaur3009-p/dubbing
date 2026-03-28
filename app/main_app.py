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
from pipelines.live_pipeline     import run_live_pipeline as run_live, reset_live
from pipelines.sentence_pipeline import run_travel_pipeline
from services.voice_identity     import is_enrolled, clear_all as clear_voice_store

import services.asr, services.asr_travel, services.translator
import services.tts_engine, services.tts_cloning
import realtime.vad, realtime.chunker, realtime.rolling_buffer

LANG_NAMES = sorted(LANGUAGES.keys())


def _f32(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.int16:
        return data.astype(np.float32) / 32768.0
    if np.issubdtype(data.dtype, np.integer):
        return data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    return data.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  Tab 1 — Streaming Dub
# ═══════════════════════════════════════════════════════════════════════════════

def _stream_cb(audio, lang, use_clone):
    set_cloning(bool(use_clone))
    if audio is None:
        return "", "", None, "", _enroll_badge()
    sr, data = audio
    t, tr, sf_path, lat = run_stream(_f32(data), sr, lang)
    return t, tr, sf_path, lat, _enroll_badge()


def _stream_reset(use_clone):
    reset_stream()
    clear_voice_store()
    set_cloning(bool(use_clone))
    return "", "", None, "Ready.", "⬜ Not enrolled"


def _enroll_badge() -> str:
    return "🟢 Voice enrolled — cloning active" if is_enrolled("stream_user") else "🔴 Not enrolled (speak ~3 s)"


def build_streaming_tab():
    gr.Markdown(
        "### 🎙 Streaming Dub\n"
        "Speak naturally. Silero VAD detects each phrase, then "
        "**ASR → Translate → TTS run in a background thread** so the UI never freezes.\n\n"
        "Translation and dubbed audio appear as each phrase completes. "
        "⏳ in the translation box means that phrase is still being processed."
    )

    with gr.Row():
        # ── Left column: controls ─────────────────────────────────────────────
        with gr.Column(scale=1, min_width=280):
            mic = gr.Audio(
                sources=["microphone"],
                streaming=True,
                label="🎤 Microphone — start speaking",
            )
            lang_dd = gr.Dropdown(
                choices=LANG_NAMES,
                value="Hindi",
                label="Target language",
            )
            clone_chk = gr.Checkbox(
                value=True,
                label="🧬 Use voice cloning",
                info="Auto-enrolls your voice after ~3 s. Uncheck for generic TTS.",
            )
            enroll_badge = gr.Textbox(
                value="🔴 Not enrolled (speak ~3 s)",
                label="Clone status",
                interactive=False,
                lines=1,
            )
            reset_btn = gr.Button("🔄 Reset", variant="secondary")

        # ── Right column: live output ─────────────────────────────────────────
        with gr.Column(scale=2):
            with gr.Row():
                transcript_box = gr.Textbox(
                    label="📝 Transcript",
                    lines=9,
                    interactive=False,
                    show_copy_button=True,
                )
                translation_box = gr.Textbox(
                    label="🌐 Translation  (⏳ = processing)",
                    lines=9,
                    interactive=False,
                    show_copy_button=True,
                )
            audio_out = gr.Audio(
                label="🔊 Dubbed audio",
                autoplay=True,
                show_download_button=True,
            )
            latency_box = gr.Textbox(
                label="⏱ Last chunk timing",
                interactive=False,
                lines=1,
            )

    mic.stream(
        fn=_stream_cb,
        inputs=[mic, lang_dd, clone_chk],
        outputs=[transcript_box, translation_box, audio_out, latency_box, enroll_badge],
    )
    reset_btn.click(
        fn=_stream_reset,
        inputs=[clone_chk],
        outputs=[transcript_box, translation_box, audio_out, latency_box, enroll_badge],
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Tab 2 — Live Mic
# ═══════════════════════════════════════════════════════════════════════════════

def _live_cb(audio, lang):
    if audio is None:
        return run_live(None, 16000, lang)
    sr, data = audio
    return run_live(_f32(data), sr, lang)


def build_live_tab():
    gr.Markdown(
        "### 📡 Live Mic\n"
        "Rolling-buffer transcription with **live partial previews** "
        "(trailing `_` = in-progress word). Commits on ≥ 500 ms silence."
    )
    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            mic      = gr.Audio(sources=["microphone"], streaming=True, label="🎤 Microphone")
            lang_dd  = gr.Dropdown(choices=LANG_NAMES, value="Hindi", label="Target language")
            reset_btn = gr.Button("🔄 Reset", variant="secondary")
        with gr.Column(scale=2):
            with gr.Row():
                t_box  = gr.Textbox(label="📝 Transcript (live)",  lines=9, interactive=False, show_copy_button=True)
                tr_box = gr.Textbox(label="🌐 Translation (live)", lines=9, interactive=False, show_copy_button=True)
            lat_box = gr.Textbox(label="⏱ Latency", interactive=False, lines=1)

    mic.stream(fn=_live_cb, inputs=[mic, lang_dd], outputs=[t_box, tr_box, lat_box])
    reset_btn.click(fn=lambda: ("", "", ""), outputs=[t_box, tr_box, lat_box])


# ═══════════════════════════════════════════════════════════════════════════════
#  Tab 3 — Sentence / File
# ═══════════════════════════════════════════════════════════════════════════════

def _sentence_cb(audio, ref_audio, lang, use_clone):
    if audio is None:
        return "", "", None, "No audio."
    sr, data = audio
    data = _f32(data)

    sid = "sentence_user"
    if use_clone and ref_audio is not None:
        ref_sr, ref_data = ref_audio
        from services.voice_identity import enroll_speaker, clear_speaker
        clear_speaker(sid)
        enroll_speaker(sid, _f32(ref_data), sr=ref_sr)

    import time, tempfile, librosa
    import soundfile as sf_lib
    from realtime.vad            import detect_speech
    from services.asr_travel     import transcribe_travel
    from services.translator     import translate
    from services.tts_cloning    import speak_as
    from services.tts_engine     import speak as speak_gen
    from services.voice_identity import is_enrolled
    from services.transcript_formatter import format_full_text

    if data.ndim > 1: data = np.mean(data, axis=1)
    data = np.clip(data.astype(np.float32), -1.0, 1.0)
    if sr != 16000:
        data = librosa.resample(data, orig_sr=sr, target_sr=16000); sr = 16000
    if len(data) < 1600:
        return "", "", None, "Audio too short."

    segs = detect_speech(data, sr)
    if not segs: return "", "", None, "No speech detected."
    speech = data[segs[0]["start"]: segs[-1]["end"]]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf_lib.write(tmp.name, speech, sr); wav = tmp.name

    t0 = time.perf_counter()
    raw, src = transcribe_travel(wav)
    asr_ms = round((time.perf_counter() - t0) * 1000)
    if not raw.strip(): return "", "", None, "No transcript."
    text = format_full_text(raw) or raw.strip()

    tgt_code = LANGUAGES.get(lang, "hin_Deva")
    t_tr = time.perf_counter()
    translated = translate(text, src, tgt_code)
    tr_ms = round((time.perf_counter() - t_tr) * 1000)

    t_tts = time.perf_counter()
    if use_clone and is_enrolled(sid):
        sf_path = speak_as(translated.strip(), sid); note = "✅ cloned"
    else:
        sf_path = speak_gen(translated.strip()); note = "❌ generic"
    tts_ms = round((time.perf_counter() - t_tts) * 1000)
    total  = round((time.perf_counter() - t0) * 1000)

    lat = f"ASR {asr_ms} ms | Translate {tr_ms} ms | TTS {tts_ms} ms | Total {total} ms | {note}"
    return text, translated, sf_path, lat


def build_sentence_tab():
    gr.Markdown(
        "### 🌍 Sentence / File Dub\n"
        "Upload or record one sentence. Optionally provide a **reference voice clip** "
        "(3–10 s) to activate zero-shot voice cloning.\n\n"
        "Pipeline: Silero VAD → Whisper-medium → NLLB → CosyVoice / Parler-TTS"
    )
    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            audio_in    = gr.Audio(sources=["microphone","upload"], type="numpy", label="🎤 Speech to dub")
            ref_audio   = gr.Audio(sources=["microphone","upload"], type="numpy", label="🧬 Reference voice (optional)")
            lang_dd     = gr.Dropdown(choices=LANG_NAMES, value="Hindi", label="Target language")
            clone_chk   = gr.Checkbox(value=True, label="🧬 Use voice cloning",
                                      info="Needs a reference clip. Falls back to generic TTS.")
            run_btn     = gr.Button("▶ Dub it", variant="primary", size="lg")
        with gr.Column(scale=2):
            t_box   = gr.Textbox(label="📝 Transcript",  lines=5, interactive=False, show_copy_button=True)
            tr_box  = gr.Textbox(label="🌐 Translation", lines=5, interactive=False, show_copy_button=True)
            aud_out = gr.Audio(label="🔊 Dubbed audio",  autoplay=True, show_download_button=True)
            lat_box = gr.Textbox(label="⏱ Latency",      interactive=False, lines=1)

    run_btn.click(fn=_sentence_cb,
                  inputs=[audio_in, ref_audio, lang_dd, clone_chk],
                  outputs=[t_box, tr_box, aud_out, lat_box])


# ═══════════════════════════════════════════════════════════════════════════════
#  App
# ═══════════════════════════════════════════════════════════════════════════════

def build_app() -> gr.Blocks:
    with gr.Blocks(title="DubYou", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🎙 DubYou — Real-Time Identity-Preserving Voice Dubbing\n"
            "> *Same voice. Different language.*\n\n"
            "**Pipeline:** Silero VAD → Whisper → NLLB / Gemini → CosyVoice / Parler-TTS  \n"
            "Set `DEEPGRAM_API_KEY` for faster ASR · `GEMINI_API_KEY` for faster translation."
        )
        with gr.Tabs():
            with gr.Tab("🎙 Streaming Dub"):   build_streaming_tab()
            with gr.Tab("📡 Live Mic"):         build_live_tab()
            with gr.Tab("🌍 Sentence / File"):  build_sentence_tab()

        gr.Markdown(
            "---\n**Languages:** " + " · ".join(LANG_NAMES) + "  \n"
            "**Voice cloning:** auto-enrolls from mic · CosyVoice → Parler-TTS fallback.  \n"
            "*All models load at startup — first run downloads weights (~2 min).*"
        )
    return demo


if __name__ == "__main__":
    build_app().launch(share=True, server_port=7860, show_error=True)
