"""
LiveKit Agent — replaces Gradio WebRTC as the media transport layer.

Install:
    pip install livekit livekit-agents

Run:
    python -m app.transport.livekit_agent start \
        --url wss://YOUR_LIVEKIT_URL \
        --api-key YOUR_KEY \
        --api-secret YOUR_SECRET

Deploy on Fly.io or AWS Wavelength for edge latency reduction.
"""
import asyncio
import numpy as np

try:
    from livekit import agents, rtc
    from livekit.agents import JobContext, WorkerOptions, cli
    _LIVEKIT_AVAILABLE = True
except ImportError:
    _LIVEKIT_AVAILABLE = False
    print("[livekit_agent] livekit-agents not installed. "
          "Install with: pip install livekit livekit-agents")

from app.pipelines.live_pipeline import run_live_pipeline, reset_live
from app.services.voice_identity import enroll_speaker, is_enrolled, clear_speaker
from app.transport.sync_protocol import make_sync_packet
import time
import uuid

DEFAULT_TARGET_LANG = "Hindi"
ENROLLMENT_MIN_SAMPLES = 16000 * 3   # 3 seconds at 16kHz


async def entrypoint(ctx: "JobContext"):
    """
    Main agent entry point. Called once per LiveKit room connection.
    """
    await ctx.connect()
    room = ctx.room
    print(f"[livekit_agent] connected to room: {room.name}")

    # Track per-participant state
    enrollment_buffers: dict[str, list[np.ndarray]] = {}

    @room.on("participant_disconnected")
    def on_disconnect(participant):
        clear_speaker(participant.identity)
        enrollment_buffers.pop(participant.identity, None)
        print(f"[livekit_agent] participant left: {participant.identity}")

    @room.on("track_subscribed")
    def on_track(track, publication, participant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.create_task(
                process_audio_track(track, participant, room, enrollment_buffers)
            )

    # Keep agent alive
    await asyncio.sleep(float("inf"))


async def process_audio_track(
    track,
    participant,
    room,
    enrollment_buffers: dict,
):
    """Process one participant's audio track end-to-end."""
    sid = participant.identity
    enrollment_buffers.setdefault(sid, [])

    audio_stream = rtc.AudioStream(track)
    async for frame_event in audio_stream:
        frame = frame_event.frame
        audio_np = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0

        # ── Voice enrollment (first 3–5 s) ──────────────────────────────────
        if not is_enrolled(sid):
            enrollment_buffers[sid].append(audio_np)
            total = sum(len(a) for a in enrollment_buffers[sid])
            if total >= ENROLLMENT_MIN_SAMPLES:
                combined = np.concatenate(enrollment_buffers[sid])
                enroll_speaker(sid, combined, sr=frame.sample_rate)
                enrollment_buffers[sid].clear()

        # ── Main pipeline ────────────────────────────────────────────────────
        t_start = time.perf_counter()
        transcript, translation, latency = run_live_pipeline(
            audio_np, frame.sample_rate, DEFAULT_TARGET_LANG
        )

        if translation:
            # Publish sync packet over DataChannel so clients can align playback
            chunk_id   = str(uuid.uuid4())
            orig_dur   = len(audio_np) / frame.sample_rate * 1000
            sync_pkt   = make_sync_packet(
                chunk_id        = chunk_id,
                original_start_ms = t_start * 1000,
                original_end_ms   = (t_start + orig_dur / 1000) * 1000,
                dubbed_duration_ms = orig_dur,   # updated after TTS
                speaker_id       = sid,
                target_lang      = DEFAULT_TARGET_LANG,
            )
            await room.local_participant.publish_data(
                sync_pkt.encode(), reliable=True
            )


if _LIVEKIT_AVAILABLE:
    if __name__ == "__main__":
        cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
