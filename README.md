# DubYou — Real-time Identity-Preserving Speech Translation

Sub-500ms conversational dubbing with voice cloning.  
**Target total latency:** δ_net(40) + δ_vad(100) + δ_stt(120) + δ_llm(80) + δ_tts(100) ≈ **440ms**

---

## Architecture

```
LiveKit SFU (Edge)
  └── Audio track per participant
        ├── Semantic VAD          ~100ms  silero_vad / Deepgram Flux
        ├── Streaming STT         ~120ms  Deepgram Nova-3
        ├── Translation           ~80ms   Gemini 1.5 Flash (NLLB fallback)
        ├── Zero-shot TTS         ~100ms  CosyVoice / Qwen3-TTS
        └── Sync packets                  WebRTC DataChannel
```

---

## Migration Phases

### Phase 1 (current codebase + voice cloning)
- Keep Gradio transport
- Add `voice_identity.py` enrollment
- Swap `tts_engine.py` → `tts_cloning.py` (CosyVoice)
- **Impact:** Same voice across languages — biggest UX improvement

### Phase 2 (streaming inference)
- Set `DEEPGRAM_API_KEY` → enables `asr_streaming.py`
- Set `GEMINI_API_KEY`   → enables `translator_gemini.py`
- Drops STT latency from ~400ms → ~120ms
- Removes local GPU requirement for STT

### Phase 3 (LiveKit transport)
- Install `livekit livekit-agents`
- Run `python -m app.transport.livekit_agent start`
- Deploy workers on Fly.io nearest your users
- Enables multi-participant rooms, per-speaker voice cloning

### Phase 4 (full sync)
- Enable `sync_protocol.py` DataChannel packets
- Integrate `client/sync_buffer.js` adaptive jitter buffer
- Add semantic VAD (`semantic_vad.py`) for early endpointing

---

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

## Running

**Current Gradio UI (no changes required):**
```bash
python app.py
```

**LiveKit agent (Phase 3):**
```bash
python -m app.transport.livekit_agent start \
  --url $LIVEKIT_URL \
  --api-key $LIVEKIT_API_KEY \
  --api-secret $LIVEKIT_API_SECRET
```

---

## File Map

| File | Status | Purpose |
|------|--------|---------|
| `app/pipelines/live_pipeline.py` | Existing | Gradio live streaming pipeline |
| `app/pipelines/sentence_pipeline.py` | Existing | Single-sentence travel pipeline |
| `app/pipelines/streaming_pipeline.py` | Existing | Chunked streaming pipeline |
| `app/realtime/rolling_buffer.py` | Existing | VAD + commit/partial logic |
| `app/realtime/parallel_worker.py` | Existing | Thread-pool chunk processor |
| `app/realtime/chunker.py` | Existing | VoiceChunker (Silero VAD) |
| `app/realtime/vad.py` | Existing | Silero VAD detect_speech() |
| `app/realtime/semantic_vad.py` | **NEW** | Semantic utterance completeness |
| `app/services/asr.py` | Existing | faster-whisper small (partial) |
| `app/services/asr_travel.py` | Existing | faster-whisper medium (commit) |
| `app/services/asr_streaming.py` | **NEW** | Deepgram Nova-3 streaming |
| `app/services/translator.py` | Existing | NLLB-200 local translation |
| `app/services/translator_gemini.py` | **NEW** | Gemini 1.5 Flash streaming |
| `app/services/tts_engine.py` | Existing | Parler TTS generic voice |
| `app/services/tts_cloning.py` | **NEW** | CosyVoice zero-shot cloning |
| `app/services/voice_identity.py` | **NEW** | Enrollment + latent cache |
| `app/services/transcript_formatter.py` | Existing | Clean + rolling window |
| `app/transport/livekit_agent.py` | **NEW** | LiveKit SFU agent worker |
| `app/transport/sync_protocol.py` | **NEW** | DataChannel sync packets |
| `client/sync_buffer.js` | **NEW** | Adaptive jitter buffer (JS) |
