"""
main_app.py  —  Real-Time Multilingual Dubbing Studio
======================================================
Three tabs, three pipelines:
  • LIVE DUB   — streaming_pipeline  (Deepgram/Whisper + Gemini/NLLB + TTS + voice clone)
  • LIVE MIC   — live_pipeline       (rolling-buffer, partial previews, no TTS)
  • SENTENCE   — sentence_pipeline   (record once, full quality translate+TTS)

Run:
    python main_app.py
    # or
    gradio main_app.py
"""

from __future__ import annotations
import os
import numpy as np
import gradio as gr

# ── Pipeline imports ──────────────────────────────────────────────────────────
from pipelines.streaming_pipeline import (
    run_pipeline,
    reset_stream,
    init_pipeline,
    set_cloning,
    LANGUAGES as STREAM_LANGS,
)
from pipelines.live_pipeline import (
    run_live_pipeline as run_live,
    reset_live,
    LANGUAGES as LIVE_LANGS,
)
from pipelines.sentence_pipeline import (
    run_travel_pipeline as run_sentence,
    LANGUAGES as SENTENCE_LANGS,
)

# ── Startup ───────────────────────────────────────────────────────────────────
init_pipeline()

LANG_NAMES  = list(STREAM_LANGS.keys())
_FLAG_MAP   = {
    "English": "🇬🇧", "Hindi": "🇮🇳", "Bengali": "🇧🇩", "Tamil": "🇮🇳",
    "Telugu": "🇮🇳",  "Kannada": "🇮🇳", "Malayalam": "🇮🇳", "Marathi": "🇮🇳",
    "Gujarati": "🇮🇳", "Punjabi": "🇮🇳", "Urdu": "🇵🇰", "Nepali": "🇳🇵",
    "Odia": "🇮🇳",    "Assamese": "🇮🇳", "Sindhi": "🇵🇰", "Sanskrit": "🕉️",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _f32(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.int16:
        return data.astype(np.float32) / 32768.0
    return data.astype(np.float32)

def _flag(lang: str) -> str:
    return _FLAG_MAP.get(lang, "🌐")


# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS  — broadcast studio dark theme
# ══════════════════════════════════════════════════════════════════════════════
CUSTOM_CSS = """
/* ── Base ─────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #0a0b0f;
    --bg2:       #12141a;
    --bg3:       #1a1d26;
    --border:    #2a2d3a;
    --accent:    #e8ff47;
    --accent2:   #47ffe8;
    --red:       #ff4747;
    --muted:     #5a5f7a;
    --text:      #e8eaf0;
    --text2:     #9095b0;
    --mono:      'Space Mono', monospace;
    --sans:      'DM Sans', sans-serif;
    --radius:    6px;
    --glow:      0 0 20px rgba(232,255,71,0.12);
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: var(--sans) !important;
    color: var(--text) !important;
}

/* ── Header banner ────────────────────────────────────────────── */
.studio-header {
    background: var(--bg2);
    border-bottom: 1px solid var(--border);
    padding: 20px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0;
}
.studio-logo {
    font-family: var(--mono);
    font-size: 22px;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -0.5px;
}
.studio-logo span {
    color: var(--text2);
    font-weight: 400;
}
.studio-tagline {
    font-size: 12px;
    color: var(--muted);
    font-family: var(--mono);
    text-transform: uppercase;
    letter-spacing: 2px;
}
.rec-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: var(--red);
    border-radius: 50%;
    margin-right: 6px;
    animation: blink 1.4s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

/* ── Tabs ─────────────────────────────────────────────────────── */
.tab-nav { background: var(--bg2) !important; border-bottom: 1px solid var(--border) !important; }
.tab-nav button {
    font-family: var(--mono) !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    color: var(--muted) !important;
    border: none !important;
    padding: 14px 24px !important;
    background: transparent !important;
}
.tab-nav button.selected {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: transparent !important;
}
.tab-nav button:hover { color: var(--text) !important; }

/* ── Cards / panels ───────────────────────────────────────────── */
.panel-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
}
.panel-label {
    font-family: var(--mono);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--muted);
    margin-bottom: 10px;
}

/* ── Gradio component overrides ───────────────────────────────── */
.gradio-container .wrap,
.gradio-container .form {
    background: var(--bg) !important;
    border: none !important;
    gap: 16px !important;
}
label.svelte-1b6s6vi, .block > label {
    font-family: var(--mono) !important;
    font-size: 10px !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    color: var(--muted) !important;
    margin-bottom: 6px !important;
}
textarea, input[type="text"] {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
    padding: 12px 14px !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(232,255,71,0.08) !important;
    outline: none !important;
}

/* transcript — monospace feel */
.transcript-box textarea {
    font-family: var(--sans) !important;
    font-size: 13.5px !important;
    color: var(--text) !important;
    min-height: 200px !important;
}
.translation-box textarea {
    font-family: var(--sans) !important;
    font-size: 13.5px !important;
    color: var(--accent2) !important;
    min-height: 200px !important;
}

/* latency / status boxes */
.latency-box textarea, .status-box textarea {
    font-family: var(--mono) !important;
    font-size: 11px !important;
    color: var(--muted) !important;
    min-height: 36px !important;
    max-height: 36px !important;
    padding: 8px 12px !important;
    border-color: transparent !important;
    background: var(--bg3) !important;
}

/* ── Buttons ──────────────────────────────────────────────────── */
button.primary {
    background: var(--accent) !important;
    color: #0a0b0f !important;
    font-family: var(--mono) !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: var(--radius) !important;
    padding: 10px 20px !important;
    cursor: pointer !important;
    transition: opacity 0.15s !important;
}
button.primary:hover { opacity: 0.88 !important; }
button.secondary {
    background: transparent !important;
    color: var(--muted) !important;
    font-family: var(--mono) !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 10px 20px !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
}
button.secondary:hover {
    border-color: var(--muted) !important;
    color: var(--text) !important;
}

/* ── Dropdown ─────────────────────────────────────────────────── */
.wrap.svelte-w3nnwh, select, .multiselect {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}

/* ── Checkbox ─────────────────────────────────────────────────── */
input[type="checkbox"] { accent-color: var(--accent) !important; }

/* ── Audio component ──────────────────────────────────────────── */
.audio-component audio {
    background: var(--bg3) !important;
    border-radius: var(--radius) !important;
    width: 100% !important;
    filter: invert(0) !important;
}
.waveform-container { background: var(--bg3) !important; border-radius: var(--radius) !important; }

/* ── Tier badge ───────────────────────────────────────────────── */
.tier-display {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    padding: 8px 0;
}

/* ── Section dividers ─────────────────────────────────────────── */
.section-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 8px 0;
}

/* ── Clone badge ──────────────────────────────────────────────── */
.clone-enrolled textarea { color: #47ff9a !important; }
.clone-pending textarea  { color: var(--red) !important; }

/* ── Output audio ─────────────────────────────────────────────── */
.dubbed-audio {
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 12px;
    background: var(--bg3);
}

/* ── Sentence result card ─────────────────────────────────────── */
.result-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
}
.result-stat {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px;
    font-family: var(--mono);
}
.result-stat-value {
    font-size: 22px;
    font-weight: 700;
    color: var(--accent);
}
.result-stat-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--muted);
    margin-top: 2px;
}

/* ── Scrollbar ────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

/* ── Tab content spacing ──────────────────────────────────────── */
.tabitem { padding: 24px !important; }

/* ── Info text ────────────────────────────────────────────────── */
.gr-prose, .gr-markdown p {
    color: var(--text2) !important;
    font-size: 13px !important;
    line-height: 1.6 !important;
}
.gr-prose code, .gr-markdown code {
    background: var(--bg3) !important;
    color: var(--accent) !important;
    font-family: var(--mono) !important;
    padding: 2px 6px !important;
    border-radius: 3px !important;
}

/* ── Pipeline info pill ───────────────────────────────────────── */
.pipeline-info {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 100px;
    padding: 6px 14px;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
"""


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — LIVE DUB  (streaming_pipeline: full chain incl. TTS + voice clone)
# ══════════════════════════════════════════════════════════════════════════════

def _stream_cb(audio, lang, clone):
    """Gradio stream callback → streaming_pipeline.run_pipeline"""
    set_cloning(clone)
    if audio is None:
        t, tr, sf_path, lat = run_pipeline(None, 16000, lang)
        enrolled = _enroll_badge(clone)
        return t, tr, sf_path, lat, enrolled
    sr, data = audio
    frame = _f32(data)
    t, tr, sf_path, lat = run_pipeline(frame, sr, lang)
    enrolled = _enroll_badge(clone)
    return t, tr, sf_path, lat, enrolled


def _enroll_badge(clone_enabled: bool) -> str:
    try:
        from services.voice_identity import is_enrolled
        if not clone_enabled:
            return "🔇 Voice cloning disabled"
        if is_enrolled("stream_user"):
            return "✅ Voice enrolled — cloning active"
        return "🔴 Not enrolled — speak ~3s to auto-enroll"
    except Exception:
        return "⚠️ Voice identity service unavailable"


def _stream_reset(clone):
    reset_stream()
    set_cloning(clone)
    return "", "", None, "", _enroll_badge(clone)


def build_live_dub_tab():
    gr.HTML("""
    <div style="margin-bottom:20px;">
        <div style="font-family:'Space Mono',monospace; font-size:11px; text-transform:uppercase;
                    letter-spacing:2px; color:#5a5f7a; margin-bottom:6px;">Pipeline</div>
        <div style="display:flex; gap:8px; flex-wrap:wrap;">
            <span class="pipeline-info">🎙 Deepgram Nova-3 <em>or</em> Whisper</span>
            <span class="pipeline-info">🌐 Gemini 1.5 Flash <em>or</em> NLLB-600M</span>
            <span class="pipeline-info">🔊 Indic Parler-TTS</span>
            <span class="pipeline-info">🧬 CosyVoice zero-shot clone</span>
        </div>
        <div style="margin-top:10px; font-size:12px; color:#5a5f7a; font-family:'DM Sans',sans-serif;">
            Speak into the mic. Your voice is transcribed, translated, and dubbed in real time
            with optional voice cloning. Set <code>DEEPGRAM_API_KEY</code> + <code>GEMINI_API_KEY</code>
            for Tier 1 (~300 ms). Without keys, falls back to Tier 3 (~800 ms).
        </div>
    </div>
    """)

    with gr.Row(equal_height=False):
        # ── LEFT: controls ──────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=260):
            gr.HTML('<div class="panel-label">Input</div>')
            mic = gr.Audio(
                sources=["microphone"],
                streaming=True,
                label="Microphone",
                elem_classes=["audio-component"],
            )

            gr.HTML('<div style="height:12px"></div>')
            gr.HTML('<div class="panel-label">Target Language</div>')
            lang_dd = gr.Dropdown(
                choices=[f"{_flag(l)}  {l}" for l in LANG_NAMES],
                value=f"{_flag('Hindi')}  Hindi",
                label="",
                show_label=False,
            )

            gr.HTML('<div style="height:8px"></div>')
            clone_chk = gr.Checkbox(
                value=True,
                label="🧬  Voice cloning  (auto-enrolls from mic)",
            )
            enroll_badge = gr.Textbox(
                value="🔴 Not enrolled — speak ~3s to auto-enroll",
                label="Clone status",
                interactive=False,
                lines=1,
                elem_classes=["status-box", "clone-pending"],
            )

            gr.HTML('<div style="height:12px"></div>')
            reset_btn = gr.Button("↺  Reset session", variant="secondary")

        # ── RIGHT: outputs ──────────────────────────────────────────────────
        with gr.Column(scale=2):
            gr.HTML('<div class="panel-label">Live Transcript  ·  Translation</div>')
            with gr.Row():
                t_box = gr.Textbox(
                    label="Transcript  (▌ = live word)",
                    lines=9,
                    interactive=False,
                    show_copy_button=True,
                    elem_classes=["transcript-box"],
                    placeholder="Transcript will appear here as you speak…",
                )
                tr_box = gr.Textbox(
                    label="Translation  (⏳ = processing)",
                    lines=9,
                    interactive=False,
                    show_copy_button=True,
                    elem_classes=["translation-box"],
                    placeholder="Dubbed translation will appear here…",
                )

            gr.HTML('<div class="panel-label" style="margin-top:16px;">Dubbed Audio Output</div>')
            audio_out = gr.Audio(
                label="",
                autoplay=True,
                show_download_button=True,
                elem_classes=["dubbed-audio"],
            )

            lat_box = gr.Textbox(
                label="Latency  ·  Pipeline status",
                interactive=False,
                lines=1,
                elem_classes=["latency-box"],
                placeholder="Timing metrics will appear after first chunk…",
            )

    # ── Wire callbacks ──────────────────────────────────────────────────────
    def _parse_lang(display_val: str) -> str:
        """Strip the flag emoji prefix from the dropdown value."""
        return display_val.split("  ", 1)[-1].strip() if "  " in display_val else display_val

    def _cb(audio, lang_display, clone):
        return _stream_cb(audio, _parse_lang(lang_display), clone)

    def _reset(clone):
        return _stream_reset(clone)

    mic.stream(
        fn=_cb,
        inputs=[mic, lang_dd, clone_chk],
        outputs=[t_box, tr_box, audio_out, lat_box, enroll_badge],
    )
    reset_btn.click(
        fn=_reset,
        inputs=[clone_chk],
        outputs=[t_box, tr_box, audio_out, lat_box, enroll_badge],
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — LIVE MIC  (live_pipeline: rolling buffer, partials, no TTS)
# ══════════════════════════════════════════════════════════════════════════════

def _live_cb(audio, lang):
    if audio is None:
        return run_live(None, 16000, lang)
    sr, data = audio
    return run_live(_f32(data), sr, lang)


def _parse_lang_live(display_val: str) -> str:
    return display_val.split("  ", 1)[-1].strip() if "  " in display_val else display_val


def build_live_tab():
    gr.HTML("""
    <div style="margin-bottom:20px;">
        <div style="font-family:'Space Mono',monospace; font-size:11px; text-transform:uppercase;
                    letter-spacing:2px; color:#5a5f7a; margin-bottom:6px;">Pipeline</div>
        <div style="display:flex; gap:8px; flex-wrap:wrap;">
            <span class="pipeline-info">🎙 Whisper medium (rolling buffer)</span>
            <span class="pipeline-info">🌐 NLLB-600M</span>
            <span class="pipeline-info">⚡ Partial previews every 600 ms</span>
        </div>
        <div style="margin-top:10px; font-size:12px; color:#5a5f7a; font-family:'DM Sans',sans-serif;">
            Continuous rolling transcription with live partial word previews.
            Translation commits on ≥ 500 ms of silence. No TTS — text only.
            Best for note-taking or captioning use cases.
        </div>
    </div>
    """)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=260):
            gr.HTML('<div class="panel-label">Input</div>')
            mic = gr.Audio(
                sources=["microphone"],
                streaming=True,
                label="Microphone",
                elem_classes=["audio-component"],
            )
            gr.HTML('<div style="height:12px"></div>')
            gr.HTML('<div class="panel-label">Target Language</div>')
            lang_dd = gr.Dropdown(
                choices=[f"{_flag(l)}  {l}" for l in LANG_NAMES],
                value=f"{_flag('Hindi')}  Hindi",
                label="",
                show_label=False,
            )
            gr.HTML('<div style="height:12px"></div>')
            reset_btn = gr.Button("↺  Reset", variant="secondary")

        with gr.Column(scale=2):
            gr.HTML('<div class="panel-label">Live Transcript  ·  Translation</div>')
            with gr.Row():
                t_box = gr.Textbox(
                    label="Transcript",
                    lines=11,
                    interactive=False,
                    show_copy_button=True,
                    elem_classes=["transcript-box"],
                    placeholder="Rolling transcript appears here…",
                )
                tr_box = gr.Textbox(
                    label="Translation",
                    lines=11,
                    interactive=False,
                    show_copy_button=True,
                    elem_classes=["translation-box"],
                    placeholder="Translation commits on silence…",
                )
            lat_box = gr.Textbox(
                label="Latency  ·  Pipeline status",
                interactive=False,
                lines=1,
                elem_classes=["latency-box"],
                placeholder="Timing will appear after first commit…",
            )

    def _cb(audio, lang_display):
        return _live_cb(audio, _parse_lang_live(lang_display))

    mic.stream(
        fn=_cb,
        inputs=[mic, lang_dd],
        outputs=[t_box, tr_box, lat_box],
    )
    reset_btn.click(
        fn=lambda: ("", "", ""),
        outputs=[t_box, tr_box, lat_box],
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — SENTENCE / FILE  (sentence_pipeline: record → translate → TTS)
# ══════════════════════════════════════════════════════════════════════════════

def _sentence_cb(audio, lang):
    """Process a recorded sentence through the full travel pipeline."""
    if audio is None:
        return "", "", None, "No audio provided."
    sr, data = audio
    arr = _f32(data)
    text, translated, speech_file, timings = run_sentence(arr, sr, lang)
    if not text:
        return "", "", None, "⚠ No speech detected. Please try again."

    timing_str = (
        f"ASR {timings.get('asr', 0):.0f} ms  ·  "
        f"Translate {timings.get('translation', 0):.0f} ms  ·  "
        f"TTS {timings.get('tts', 0):.0f} ms  ·  "
        f"Total {timings.get('total', 0):.0f} ms"
    )
    return text, translated, speech_file, timing_str


def build_sentence_tab():
    gr.HTML("""
    <div style="margin-bottom:20px;">
        <div style="font-family:'Space Mono',monospace; font-size:11px; text-transform:uppercase;
                    letter-spacing:2px; color:#5a5f7a; margin-bottom:6px;">Pipeline</div>
        <div style="display:flex; gap:8px; flex-wrap:wrap;">
            <span class="pipeline-info">🎙 Whisper medium (beam=5, VAD filter)</span>
            <span class="pipeline-info">🌐 NLLB-600M</span>
            <span class="pipeline-info">🔊 Indic Parler-TTS</span>
        </div>
        <div style="margin-top:10px; font-size:12px; color:#5a5f7a; font-family:'DM Sans',sans-serif;">
            Record a sentence or upload an audio file. The full pipeline runs once with
            maximum quality settings — higher accuracy than the live modes, best for
            document translation or reviewing a specific phrase.
        </div>
    </div>
    """)

    with gr.Row(equal_height=False):
        # ── LEFT ────────────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=260):
            gr.HTML('<div class="panel-label">Input Audio</div>')
            mic = gr.Audio(
                sources=["microphone", "upload"],
                type="numpy",
                label="Record or upload",
                elem_classes=["audio-component"],
            )
            gr.HTML('<div style="height:12px"></div>')
            gr.HTML('<div class="panel-label">Target Language</div>')
            lang_dd = gr.Dropdown(
                choices=[f"{_flag(l)}  {l}" for l in LANG_NAMES],
                value=f"{_flag('Hindi')}  Hindi",
                label="",
                show_label=False,
            )
            gr.HTML('<div style="height:12px"></div>')
            run_btn = gr.Button("▶  Translate & Dub", variant="primary")
            clear_btn = gr.Button("✕  Clear", variant="secondary")

        # ── RIGHT ────────────────────────────────────────────────────────────
        with gr.Column(scale=2):
            gr.HTML('<div class="panel-label">Transcript  ·  Translation</div>')
            with gr.Row():
                t_box = gr.Textbox(
                    label="Original transcript",
                    lines=6,
                    interactive=False,
                    show_copy_button=True,
                    elem_classes=["transcript-box"],
                    placeholder="Transcript will appear after processing…",
                )
                tr_box = gr.Textbox(
                    label="Translation",
                    lines=6,
                    interactive=False,
                    show_copy_button=True,
                    elem_classes=["translation-box"],
                    placeholder="Translation will appear after processing…",
                )

            gr.HTML('<div class="panel-label" style="margin-top:16px;">Dubbed Audio</div>')
            audio_out = gr.Audio(
                label="",
                autoplay=False,
                show_download_button=True,
                elem_classes=["dubbed-audio"],
            )

            timing_box = gr.Textbox(
                label="Pipeline timing",
                interactive=False,
                lines=1,
                elem_classes=["latency-box"],
                placeholder="Timing breakdown will appear after processing…",
            )

    def _parse_lang_s(display_val: str) -> str:
        return display_val.split("  ", 1)[-1].strip() if "  " in display_val else display_val

    run_btn.click(
        fn=lambda audio, lang_d: _sentence_cb(audio, _parse_lang_s(lang_d)),
        inputs=[mic, lang_dd],
        outputs=[t_box, tr_box, audio_out, timing_box],
    )
    clear_btn.click(
        fn=lambda: (None, "", "", None, ""),
        outputs=[mic, t_box, tr_box, audio_out, timing_box],
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

with gr.Blocks(
    title="Multilingual Dubbing Studio",
    css=CUSTOM_CSS,
    theme=gr.themes.Base(
        primary_hue="yellow",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("DM Sans"), "sans-serif"],
        font_mono=[gr.themes.GoogleFont("Space Mono"), "monospace"],
    ),
) as demo:

    # ── Header ──────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="studio-header">
        <div>
            <div class="studio-logo">
                <span class="rec-dot"></span>
                DUB<span>/</span>STUDIO
            </div>
            <div class="studio-tagline">Real-Time Multilingual Dubbing · 16 Languages</div>
        </div>
        <div style="display:flex; gap:24px; align-items:center;">
            <div style="text-align:right;">
                <div style="font-family:'Space Mono',monospace; font-size:10px;
                            text-transform:uppercase; letter-spacing:1.5px; color:#5a5f7a;">
                    Tier 1 &nbsp;·&nbsp; ~300 ms
                </div>
                <div style="font-family:'Space Mono',monospace; font-size:10px; color:#2a2d3a;">
                    Deepgram Nova-3 + Gemini 1.5 Flash
                </div>
            </div>
            <div style="width:1px; height:36px; background:#2a2d3a;"></div>
            <div style="text-align:right;">
                <div style="font-family:'Space Mono',monospace; font-size:10px;
                            text-transform:uppercase; letter-spacing:1.5px; color:#5a5f7a;">
                    Tier 3 &nbsp;·&nbsp; ~800 ms
                </div>
                <div style="font-family:'Space Mono',monospace; font-size:10px; color:#2a2d3a;">
                    Whisper small + NLLB-600M
                </div>
            </div>
        </div>
    </div>
    """)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    with gr.Tabs(elem_classes=["tab-nav"]):

        with gr.TabItem("⬤  Live Dub", elem_classes=["tabitem"]):
            build_live_dub_tab()

        with gr.TabItem("◎  Live Mic", elem_classes=["tabitem"]):
            build_live_tab()

        with gr.TabItem("▷  Sentence / File", elem_classes=["tabitem"]):
            build_sentence_tab()

    # ── Footer ───────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="border-top: 1px solid #2a2d3a; margin-top: 24px; padding: 16px 32px;
                display: flex; justify-content: space-between; align-items: center;">
        <div style="font-family:'Space Mono',monospace; font-size:10px;
                    text-transform:uppercase; letter-spacing:1.5px; color:#2a2d3a;">
            DUB/STUDIO · Powered by Whisper · Deepgram · NLLB · Gemini · Parler-TTS · CosyVoice
        </div>
        <div style="font-family:'Space Mono',monospace; font-size:10px; color:#2a2d3a;">
            Set DEEPGRAM_API_KEY + GEMINI_API_KEY for Tier 1 performance
        </div>
    </div>
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=bool(os.getenv("GRADIO_SHARE", "")),
        show_error=True,
    )
