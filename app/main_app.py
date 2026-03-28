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
    init_pipeline,
    LANGUAGES,
    _tier_label,
    DEEPGRAM_KEY,
    GEMINI_KEY,
)
from pipelines.live_pipeline     import run_live_pipeline as run_live, reset_live
from pipelines.sentence_pipeline import run_travel_pipeline
from services.voice_identity     import is_enrolled, clear_all as clear_voice_store

import services.asr, services.asr_travel, services.translator
import services.tts_engine, services.tts_cloning
import realtime.vad, realtime.chunker, realtime.rolling_buffer

# ── Startup: load fast services and start Deepgram if key present ─────────────
init_pipeline()

LANG_NAMES = sorted(LANGUAGES.keys())


def _f32(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.int16:
        return data.astype(np.float32) / 32768.0
    if np.issubdtype(data.dtype, np.integer):
        return data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    return data.astype(np.float32)


def _enroll_badge() -> str:
    return "🟢 Voice enrolled — cloning active" if is_enrolled("stream_user") \
           else "🔴 Not enrolled (speak ~3 s to auto-enroll)"


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
    return "", "", None, "Session reset.", "🔴 Not enrolled"


def build_streaming_tab():
    # Determine current tier for the info banner
    if DEEPGRAM_KEY and GEMINI_KEY:
        tier_info = (
            "🟢 **Tier 1 active — Deepgram + Gemini (~300 ms)**\n"
            "Deepgram streams words live as you speak. "
            "Gemini translates each utterance in ~80 ms."
        )
    elif DEEPGRAM_KEY:
        tier_info = (
            "🟡 **Tier 2 active — Deepgram + NLLB (~500 ms)**\n"
            "Deepgram streams live ASR. Set `GEMINI_API_KEY` for faster translation."
        )
    else:
        tier_info = (
            "🔴 **Tier 3 active — Whisper + NLLB (~800 ms)**\n"
            "No API keys set. Set `DEEPGRAM_API_KEY` for ~3× faster ASR. "
            "Set `GEMINI_API_KEY` for ~4× faster translation."
        )

    gr.Markdown(f"### 🎙 Streaming Dub\n\n{tier_info}")

    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            mic = gr.Audio(
                sources=["microphone"], streaming=True,
                label="🎤 Microphone"
            )
            lang_dd = gr.Dropdown(
                choices=LANG_NAMES, value="Hindi", label="Target language"
            )
            clone_chk = gr.Checkbox(
                value=True,
                label="🧬 Use voice cloning",
                info="Auto-enrolls from your mic after ~3 s. Uncheck for generic TTS.",
            )
            enroll_badge = gr.Textbox(
                value="🔴 Not enrolled (speak ~3 s to auto-enroll)",
                label="Clone status", interactive=False, lines=1,
            )
            reset_btn = gr.Button("🔄 Reset session", variant="secondary")

        with gr.Column(scale=2):
            with gr.Row():
                t_box = gr.Textbox(
                    label="📝 Transcript  (▌ = live word from Deepgram)",
                    lines=9, interactive=False, show_copy_button=True,
                )
                tr_box = gr.Textbox(
                    label="🌐 Translation  (⏳ = processing)",
                    lines=9, interactive=False, show_copy_button=True,
                )
            audio_out = gr.Audio(
                label="🔊 Dubbed audio", autoplay=True, show_download_button=True,
            )
            lat_box = gr.Textbox(
                label="⏱ Last chunk timing",
                interactive=False, lines=1,
            )

    mic.stream(
        fn=_stream_cb,
        inputs=[mic, lang_dd, clone_chk],
        outputs=[t_box, tr_box, audio_out, lat_box, enroll_badge],
    )
    reset_btn.click(
        fn=_stream_reset,
        inputs=[clone_chk],
        outputs=[t_box, tr_box, audio_out, lat_box, enroll_badge],
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
        "Rolling-buffer transcription with **live partial previews**. "
        "Translation commits on ≥ 500 ms silence."
    )
    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            mic      = gr.Audio(sources=["microphone"], streaming=True, label="🎤 Microphone")
            lang_dd  = gr.Dropdown(choices=LANG_NAMES, value="Hindi", label="Target language")
            reset_btn = gr.Button("🔄 Reset", variant="secondary")
        with gr.Column(scale=2):
            with gr.Row():
                t_box  = gr.Textbox(label="📝 Transcript",  lines=9, interactive=False, show_copy_button=True)
                tr_box = gr.Textbox(label="🌐 Translation", lines=9, interactive=False, show_copy_button=True)
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

    return text, translated, sf_path, \
           f"ASR {asr_ms}ms | Translate {tr_ms}ms | TTS {tts_ms}ms | Total {total}ms | {note}"


def build_sentence_tab():
    gr.Markdown(
        "### 🌍 Sentence / File Dub\n"
        "Record or upload one sentence. Add a **reference voice clip** (3–10 s) "
        "to activate zero-shot voice cloning.\n\n"
        "Pipeline: Silero VAD → Whisper-medium → NLLB → CosyVoice / Parler-TTS"
    )
    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            audio_in   = gr.Audio(sources=["microphone","upload"], type="numpy", label="🎤 Speech to dub")
            ref_audio  = gr.Audio(sources=["microphone","upload"], type="numpy", label="🧬 Reference voice (optional)")
            lang_dd    = gr.Dropdown(choices=LANG_NAMES, value="Hindi", label="Target language")
            clone_chk  = gr.Checkbox(value=True, label="🧬 Use voice cloning",
                                     info="Needs a reference clip above.")
            run_btn    = gr.Button("▶ Dub it", variant="primary", size="lg")
        with gr.Column(scale=2):
            t_box   = gr.Textbox(label="📝 Transcript",  lines=5, interactive=False, show_copy_button=True)
            tr_box  = gr.Textbox(label="🌐 Translation", lines=5, interactive=False, show_copy_button=True)
            aud_out = gr.Audio(label="🔊 Dubbed audio",  autoplay=True, show_download_button=True)
            lat_box = gr.Textbox(label="⏱ Latency",      interactive=False, lines=1)

    run_btn.click(fn=_sentence_cb,
                  inputs=[audio_in, ref_audio, lang_dd, clone_chk],
                  outputs=[t_box, tr_box, aud_out, lat_box])


# ═══════════════════════════════════════════════════════════════════════════════
#  Tab 4 — LiveKit Setup Guide
# ═══════════════════════════════════════════════════════════════════════════════

def build_livekit_tab():
    gr.Markdown("""
### 🔴 LiveKit Real-Time Mode

LiveKit enables **true sub-500ms real-time dubbing** with:
- Audio streamed over WebRTC (no Gradio polling latency)
- Dubbed audio published back directly into the room
- Multi-participant support with per-speaker voice cloning
- Adaptive jitter buffer on the client side

---

#### Step 1 — Get a free LiveKit Cloud account
Go to **https://cloud.livekit.io** → create a project → copy your:
- Server URL (e.g. `wss://yourproject.livekit.cloud`)
- API Key
- API Secret

---

#### Step 2 — Add to Colab Cell 4 (env vars)

```python
os.environ["LIVEKIT_URL"]        = "wss://yourproject.livekit.cloud"
os.environ["LIVEKIT_API_KEY"]    = "APIxxxxxxxx"
os.environ["LIVEKIT_API_SECRET"] = "your-secret"
```

---

#### Step 3 — Install and run the agent in a new Colab cell

```python
!pip install -q livekit livekit-agents

import subprocess, sys
subprocess.Popen([
    sys.executable,
    "/content/dubbing/app/transport/livekit_agent.py",
    "start"
])
print("LiveKit agent started in background")
```

---

#### Step 4 — Join a room
Open **https://meet.livekit.io**, connect to your server URL with any API key/secret.
The DubYou agent joins automatically and starts dubbing in real time.

---

#### Latency comparison

| Mode | ASR | Translate | Total |
|---|---|---|---|
| LiveKit + Deepgram + Gemini | ~120ms | ~80ms | **~350ms** |
| Gradio + Deepgram + Gemini | ~120ms | ~80ms | **~500ms** |
| Gradio + Whisper + NLLB | ~400ms | ~300ms | **~900ms** |

The difference is the **transport layer** — LiveKit streams audio continuously via WebRTC 
whereas Gradio batches frames and has polling overhead.
""")


# ═══════════════════════════════════════════════════════════════════════════════
#  Build & launch
# ═══════════════════════════════════════════════════════════════════════════════

def build_app() -> gr.Blocks:
    with gr.Blocks(title="DubYou", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🎙 DubYou — Real-Time Identity-Preserving Voice Dubbing\n"
            "> *Same voice. Different language.*\n\n"
            f"**Active tier:** {_tier_label()}\n\n"
            "Set `DEEPGRAM_API_KEY` · `GEMINI_API_KEY` · `LIVEKIT_URL` in Cell 4 for best performance."
        )
        with gr.Tabs():
            with gr.Tab("🎙 Streaming Dub"):   build_streaming_tab()
            with gr.Tab("📡 Live Mic"):         build_live_tab()
            with gr.Tab("🌍 Sentence / File"):  build_sentence_tab()
            with gr.Tab("🔴 LiveKit Setup"):    build_livekit_tab()

        gr.Markdown(
            "---\n**Languages:** " + " · ".join(LANG_NAMES) + "  \n"
            "**Voice cloning:** auto-enrolls from mic · CosyVoice → Parler-TTS fallback."
        )
    return demo


if __name__ == "__main__":
    build_app().launch(share=True, server_port=7860, show_error=True)
