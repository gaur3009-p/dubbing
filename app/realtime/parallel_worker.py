from __future__ import annotations
import tempfile
import soundfile as sf
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, Any

_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="chunk_worker")

class ParallelChunkProcessor:
    """
    Submit chunks for parallel inference and drain completed results in order.
    """
    def __init__(
        self,
        process_fn: Callable[[str], Any],
        max_in_flight: int | None = 8,
    ):
        self._fn = process_fn
        self._max = max_in_flight
        self._queue: deque[tuple[int, Future]] = deque()
        self._seq = 0

    def reset(self):
        while self._queue:
            _, fut = self._queue.popleft()
            fut.cancel()
        self._seq = 0

    def push(self, audio_chunk, sample_rate: int = 16000) -> None:
        if self._max is not None:
            while len(self._queue) >= self._max:
                if self._queue[0][1].done():
                    break
                time.sleep(0.005)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_chunk, sample_rate)
            path = tmp.name
        seq = self._seq
        self._seq += 1
        fut = _POOL.submit(self._fn, path)
        self._queue.append((seq, fut))

    def drain(self) -> list[tuple[int, Any]]:
        results = []
        while self._queue:
            seq, fut = self._queue[0]
            if not fut.done():
                break
            self._queue.popleft()
            exc = fut.exception()
            if exc is not None:
                print(f"[ParallelChunkProcessor] chunk {seq} failed: {exc}")
                continue
            results.append((seq, fut.result()))
        return results

    @property
    def in_flight(self) -> int:
        return len(self._queue)
