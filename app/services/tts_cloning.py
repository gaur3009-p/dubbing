"""
Zero-shot voice cloning TTS.
Uses CosyVoice (open-source, available now) as the implementation bridge
until Qwen3-TTS / Phantom X 3.2 are publicly accessible.

Install: pip install cosyvoice (see https://github.com/FunAudioLLM/CosyVoice)

Falls back to the generic Parler-TTS (tts_engine.py) if:
  - CosyVoice is not installed
  - The speaker has not been enrolled yet
"""
import uuid
import numpy as np
import soundfile as sf

from services.voice_identity import get_latent
from services.tts_engine import speak as speak_generic   # fallback

try:
    from cosyvoice.cli.cosyvoice import CosyVoice as _CosyVoiceClass
    _cosyvoice = _CosyVoiceClass("iic/CosyVoice-300M-SFT")
    _COSYVOICE_AVAILABLE = True
except ImportError:
    _cosyvoice = None
    _COSYVOICE_AVAILABLE = False
    print("[tts_cloning] CosyVoice not installed — using generic TTS fallback.")


def speak_as(text: str, speaker_id: str) -> str:
    """
    Synthesise `text` in the target language using the enrolled speaker's voice.

    Args:
        text:       Translated text to synthesise.
        speaker_id: Participant identifier from the transport layer.

    Returns:
        Path to the output WAV file.
    """
    latent = get_latent(speaker_id)

    if _COSYVOICE_AVAILABLE and latent is not None:
        try:
            output = _cosyvoice.inference_zero_shot(
                text,
                prompt_speech_16k=latent,
            )
            file = f"speech_{uuid.uuid4()}.wav"
            sf.write(file, output["tts_speech"].numpy(),
                     _cosyvoice.sample_rate)
            return file
        except Exception as e:
            print(f"[tts_cloning] zero-shot failed for {speaker_id!r}: {e} — falling back")

    # Generic fallback
    return speak_generic(text)
