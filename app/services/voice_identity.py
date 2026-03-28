"""
Voice enrollment and latent caching for identity-preserving TTS.

Workflow:
  1. When a participant joins and speaks their first 3–5 seconds,
     call enroll_speaker(speaker_id, audio_np).
  2. At TTS time, call get_latent(speaker_id) and pass to tts_cloning.speak_as().

The latent is currently the raw enrollment audio clip (float32, 16kHz).
Swap the store value for a speaker embedding tensor once you integrate
Qwen3-TTS or CosyVoice's encoder.
"""
import numpy as np
from collections import defaultdict
from threading import Lock

_store: dict[str, np.ndarray] = {}
_lock  = Lock()

ENROLLMENT_SECONDS = 5   # capture up to 5s of the speaker's voice


def enroll_speaker(speaker_id: str, audio_np: np.ndarray, sr: int = 16000) -> None:
    """
    Cache a voice latent for speaker_id.
    Called once when the participant first speaks.
    Thread-safe.
    """
    clip = audio_np[: sr * ENROLLMENT_SECONDS].astype(np.float32)
    with _lock:
        # Only enroll once per session — don't overwrite a good latent
        if speaker_id not in _store:
            _store[speaker_id] = clip
            print(f"[voice_identity] enrolled speaker {speaker_id!r} "
                  f"({len(clip)/sr:.2f}s)")


def get_latent(speaker_id: str) -> np.ndarray | None:
    """Return the cached enrollment clip, or None if not yet enrolled."""
    return _store.get(speaker_id)


def is_enrolled(speaker_id: str) -> bool:
    return speaker_id in _store


def clear_speaker(speaker_id: str) -> None:
    """Remove a speaker's latent (call on participant disconnect)."""
    with _lock:
        _store.pop(speaker_id, None)


def clear_all() -> None:
    with _lock:
        _store.clear()
