import torch
import numpy as np
from silero_vad import load_silero_vad, get_speech_timestamps

_model = load_silero_vad()

def detect_speech(audio: np.ndarray, sample_rate: int):
    """Return Silero speech timestamps for the given audio."""
    if isinstance(audio, np.ndarray):
        audio = audio.astype(np.float32)
    if not torch.is_tensor(audio):
        audio = torch.from_numpy(audio)
    _model.reset_states()
    return get_speech_timestamps(audio, _model, sampling_rate=sample_rate)
