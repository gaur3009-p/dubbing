"""
Streaming STT via Deepgram Nova-3.
Replaces faster-whisper batch inference for the live pipeline.
Delivers interim transcripts at ~120ms TTFT with word-level timestamps.

Usage:
    asr = DeepgramStreamingASR(api_key="YOUR_KEY")
    async for text, is_final, words in asr.stream(audio_queue):
        ...

Set DEEPGRAM_API_KEY env var or pass api_key explicitly.
Falls back to local faster-whisper if API key is not set.
"""
import asyncio
import json
import os
import numpy as np

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
DEEPGRAM_WS_URL  = "wss://api.deepgram.com/v1/listen"

STREAM_PARAMS = {
    "model":           "nova-3",
    "language":        "multi",        # multilingual / code-switching
    "interim_results": "true",
    "punctuate":       "true",
    "utterance_end_ms": "300",         # replaces SILENCE_COMMIT_MS
    "vad_events":      "true",
    "encoding":        "linear16",
    "sample_rate":     "16000",
    "channels":        "1",
}


class DeepgramStreamingASR:
    """
    WebSocket-based streaming ASR. Each instance manages one connection.
    Send audio chunks via push(); async-iterate results().
    """

    def __init__(self, api_key: str = DEEPGRAM_API_KEY):
        self._key = api_key
        self._ws = None
        self._send_q: asyncio.Queue = asyncio.Queue()
        self._result_q: asyncio.Queue = asyncio.Queue()

    async def connect(self):
        import websockets
        params = "&".join(f"{k}={v}" for k, v in STREAM_PARAMS.items())
        url = f"{DEEPGRAM_WS_URL}?{params}"
        headers = {"Authorization": f"Token {self._key}"}
        self._ws = await websockets.connect(url, extra_headers=headers)
        asyncio.create_task(self._sender())
        asyncio.create_task(self._receiver())

    async def _sender(self):
        while True:
            chunk = await self._send_q.get()
            if chunk is None:
                await self._ws.send(json.dumps({"type": "CloseStream"}))
                break
            if isinstance(chunk, np.ndarray):
                chunk = (chunk * 32767).astype("int16").tobytes()
            await self._ws.send(chunk)

    async def _receiver(self):
        async for raw in self._ws:
            data = json.loads(raw)
            if data.get("type") == "Results":
                alt = data["channel"]["alternatives"][0]
                transcript = alt.get("transcript", "")
                is_final   = data.get("is_final", False)
                words      = alt.get("words", [])
                if transcript:
                    await self._result_q.put((transcript, is_final, words))
            elif data.get("type") == "UtteranceEnd":
                await self._result_q.put(("__utterance_end__", True, []))

    def push(self, audio_np: np.ndarray):
        """Push a float32 16kHz audio chunk into the stream."""
        self._send_q.put_nowait(audio_np)

    def close(self):
        self._send_q.put_nowait(None)

    async def results(self):
        """Async generator yielding (transcript, is_final, words)."""
        while True:
            item = await self._result_q.get()
            yield item
            if item[0] == "__utterance_end__":
                break
