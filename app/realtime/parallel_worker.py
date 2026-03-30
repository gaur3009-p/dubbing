from __future__ import annotations
import tempfile
import soundfile as sf
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, Any

_POOL = ThreadPoolExecutor(max_workers=6, thread_name_prefix="chunk_worker")


class ParallelChunkProcessor:
    """
    Submit audio chunks for parallel inference; drain completed results.
    """
    def __init__(
        self,
        process_fn: Callable[[str], Any],
        max_in_flight: int | None = 8,
    ):
        self._fn  = process_fn
        self._max = max_in_flight
        self._queue: deque[tuple[int, Future]] = deque()
        self._seq  = 0

    # ── reset ──────────────────────────────────────────────────────────────
    def reset(self):
        while self._queue:
            _, fut = self._queue.popleft()
            fut.cancel()
        self._seq = 0

    # ── submit ─────────────────────────────────────────────────────────────
    def push(self, audio_chunk, sample_rate: int = 16000) -> None:
        # Back-pressure: wait only if queue is truly full AND front isn't done
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

    # ── drain_ready (non-blocking, skips slow middle chunks) ───────────────
    def drain_ready(self) -> list[tuple[int, Any]]:
        """
        Return ALL completed chunks in submission order, skipping any chunk
        that is still running.  Skipped chunks stay in the queue.
        """
        results   = []
        remaining = deque()
        while self._queue:
            seq, fut = self._queue.popleft()
            if not fut.done():
                remaining.append((seq, fut))
                continue
            exc = fut.exception()
            if exc is not None:
                print(f"[ParallelChunkProcessor] chunk {seq} failed: {exc}")
                continue
            results.append((seq, fut.result()))
        for item in remaining:
            self._queue.appendleft(item)
        self._queue = deque(sorted(self._queue, key=lambda x: x[0]))
        return results

    # ── drain (strict-order) ───────────────────────────────────────────────
    def drain(self) -> list[tuple[int, Any]]:
        """
        Strict-order drain: stops at the first unfinished chunk.
        Kept for callers that need guaranteed ordering (sentence pipeline).
        """
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
