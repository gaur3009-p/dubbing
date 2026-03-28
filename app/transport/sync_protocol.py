"""
Sync packet protocol for the WebRTC DataChannel.

The receiver uses these packets to:
  - Align the dubbed audio track with the original timeline.
  - Suppress (duck) the original audio during dubbed playback.
  - Drive the adaptive jitter buffer on the client side.

Packet schema (JSON, v1):
  {
    "v":                 1,
    "chunk_id":          "<uuid>",
    "ts":                <unix float>,
    "orig_start_ms":     <float>,
    "orig_end_ms":       <float>,
    "dubbed_duration_ms":<float>,
    "speaker_id":        "<str>",
    "target_lang":       "<str>"
  }
"""
import json
import time
import uuid as _uuid


def make_sync_packet(
    chunk_id: str,
    original_start_ms: float,
    original_end_ms: float,
    dubbed_duration_ms: float,
    speaker_id: str,
    target_lang: str,
) -> str:
    """
    Serialise a sync packet to JSON for DataChannel transmission.
    """
    return json.dumps({
        "v":                  1,
        "chunk_id":           chunk_id,
        "ts":                 time.time(),
        "orig_start_ms":      round(original_start_ms, 2),
        "orig_end_ms":        round(original_end_ms, 2),
        "dubbed_duration_ms": round(dubbed_duration_ms, 2),
        "speaker_id":         speaker_id,
        "target_lang":        target_lang,
    })


def parse_sync_packet(raw: str | bytes) -> dict:
    """Deserialise and validate a sync packet."""
    if isinstance(raw, bytes):
        raw = raw.decode()
    pkt = json.loads(raw)
    assert pkt.get("v") == 1, f"Unknown sync packet version: {pkt.get('v')}"
    return pkt
