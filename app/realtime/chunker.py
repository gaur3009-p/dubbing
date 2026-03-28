import numpy as np
import torch
from silero_vad import load_silero_vad, get_speech_timestamps

_vad_model = load_silero_vad()

SAMPLE_RATE        = 16000
MIN_SPEECH_MS      = 300
SILENCE_TRIGGER_MS = 500
MAX_CHUNK_MS       = 8000

class VoiceChunker:
    def __init__(
        self,
        silence_trigger_ms: int = SILENCE_TRIGGER_MS,
        min_speech_ms: int = MIN_SPEECH_MS,
        max_chunk_ms: int = MAX_CHUNK_MS,
    ):
        self.silence_trigger = int(silence_trigger_ms * SAMPLE_RATE / 1000)
        self.min_speech      = int(min_speech_ms      * SAMPLE_RATE / 1000)
        self.max_chunk       = int(max_chunk_ms        * SAMPLE_RATE / 1000)
        self._buf: np.ndarray = np.array([], dtype=np.float32)
        self._speech_start: int | None = None
        self._silence_since: int = 0

    def reset(self):
        self._buf = np.array([], dtype=np.float32)
        self._speech_start = None
        self._silence_since = 0

    def push(self, audio: np.ndarray, sr: int) -> np.ndarray | None:
        import librosa
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32)
        audio = np.clip(audio, -1.0, 1.0)
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        self._buf = np.concatenate([self._buf, audio])
        timestamps = _get_timestamps(self._buf)
        if not timestamps:
            tail = min(len(self._buf), self.silence_trigger)
            self._buf = self._buf[-tail:]
            self._speech_start  = None
            self._silence_since = 0
            return None
        last_ts = timestamps[-1]
        speech_end_sample = last_ts["end"]
        if self._speech_start is None:
            self._speech_start = timestamps[0]["start"]
        samples_after_speech = len(self._buf) - speech_end_sample
        speech_len = speech_end_sample - self._speech_start
        flush = False
        if samples_after_speech >= self.silence_trigger and speech_len >= self.min_speech:
            flush = True
        elif speech_len >= self.max_chunk:
            flush = True
        if flush:
            chunk = self._buf[self._speech_start: speech_end_sample].copy()
            self._buf = self._buf[speech_end_sample:]
            self._speech_start  = None
            self._silence_since = 0
            return chunk
        return None

def _get_timestamps(audio: np.ndarray):
    t = torch.from_numpy(audio)
    _vad_model.reset_states()
    return get_speech_timestamps(t, _vad_model, sampling_rate=SAMPLE_RATE)
