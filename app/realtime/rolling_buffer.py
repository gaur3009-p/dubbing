import numpy as np
import librosa
import torch
from silero_vad import load_silero_vad, get_speech_timestamps

SAMPLE_RATE = 16000

_vad = load_silero_vad()


def _resample(audio: np.ndarray, sr: int) -> np.ndarray:
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    else:
        audio = audio.astype(np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio


def _vad_ts(audio: np.ndarray):
    _vad.reset_states()
    return get_speech_timestamps(
        torch.from_numpy(audio), _vad, sampling_rate=SAMPLE_RATE
    )


class RollingBuffer:
    PARTIAL_INTERVAL_MS = 600
    SILENCE_COMMIT_MS   = 500
    MIN_SPEECH_MS       = 150
    MAX_CHUNK_MS        = 8000

    def __init__(self):
        self._buf: np.ndarray  = np.array([], dtype=np.float32)
        self._speech_start: int | None = None
        self._new_since_partial: int = 0

    def reset(self):
        self._buf               = np.array([], dtype=np.float32)
        self._speech_start      = None
        self._new_since_partial = 0

    def push(self, audio: np.ndarray, sr: int):
        audio = _resample(audio, sr)
        n_new = len(audio)
        self._buf = np.concatenate([self._buf, audio])

        timestamps = _vad_ts(self._buf)

        if not timestamps:
            tail = int(self.SILENCE_COMMIT_MS * SAMPLE_RATE / 1000)
            self._buf               = self._buf[-tail:]
            self._speech_start      = None
            self._new_since_partial = 0
            return None, None

        first_start = timestamps[0]["start"]
        last_end    = timestamps[-1]["end"]

        if self._speech_start is None:
            self._speech_start = first_start
            self._new_since_partial = 0

        self._new_since_partial += n_new

        silence_samples  = len(self._buf) - last_end
        speech_samples   = last_end - self._speech_start
        partial_threshold = int(self.PARTIAL_INTERVAL_MS * SAMPLE_RATE / 1000)
        silence_threshold = int(self.SILENCE_COMMIT_MS   * SAMPLE_RATE / 1000)
        min_speech        = int(self.MIN_SPEECH_MS        * SAMPLE_RATE / 1000)
        max_chunk         = int(self.MAX_CHUNK_MS          * SAMPLE_RATE / 1000)

        if (silence_samples >= silence_threshold and speech_samples >= min_speech) \
                or speech_samples >= max_chunk:
            commit = self._buf[self._speech_start: last_end].copy()
            self._buf               = self._buf[last_end:]
            self._speech_start      = None
            self._new_since_partial = 0
            return None, commit

        if speech_samples >= min_speech \
                and self._new_since_partial >= partial_threshold:
            partial = self._buf[self._speech_start: last_end].copy()
            self._new_since_partial = 0
            return partial, None

        return None, None
