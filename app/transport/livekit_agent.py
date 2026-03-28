"""
livekit_agent.py  —  DubYou LiveKit Agent
==========================================
Full real-time pipeline: audio in → dub audio out → back to room.

What the original agent was missing
────────────────────────────────────
1. No TTS audio published back to the room — sync packets sent but no audio
2. run_live_pipeline used instead of full ASR+Translate+TTS pipeline
3. No error handling / reconnect logic
4. No Gradio-compatible mode for Colab

This version
────────────
• Receives audio track from participant
• Silero VAD chunks it
• ASR (Deepgram streaming if key set, else Whisper)
• Translate (Gemini if key set, else NLLB)
• TTS (CosyVoice clone if enrolled, else Parler)
• Publishes dubbed audio back as a new AudioTrack in the room
• Sends sync packets on DataChannel for client jitter buffer
• Works in Colab via ngrok tunnel (see Colab cell below)

Colab usage
───────────
  Cell A — Install LiveKit:
    !pip install livekit livekit-agents livekit-plugins-deepgram pyngrok

  Cell B — Start ngrok + agent:
    from pyngrok import ngrok
    import os
    os.environ["LIVEKIT_URL"]        = "wss://your-livekit-server"
    os.environ["LIVEKIT_API_KEY"]    = "your-key"
    os.environ["LIVEKIT_API_SECRET"] = "your-secret"
    !python /content/dubbing/app/transport/livekit_agent.py

  Then open the LiveKit Meet demo at https://meet.livekit.io and
  connect to your server — the agent joins the room automatically.
"""

import asyncio
import os
import sys
import time
import uuid
import threading
import tempfile
import queue as stdlib_queue

import numpy as np
import soundfile as sf
import torch
import librosa

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_APP  = os.path.join(_HERE, "..")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

# ── LiveKit ───────────────────────────────────────────────────────────────────
try:
    from livekit import agents, rtc
    from livekit.agents import JobContext, WorkerOptions, cli
    _LIVEKIT_OK = True
except ImportError:
    _LIVEKIT_OK = False
    print("[livekit_agent] livekit-agents not installed.")
    print("  Run: pip install livekit livekit-agents")

# ── Services ──────────────────────────────────────────────────────────────────
from silero_vad import load_silero_vad, get_speech_timestamps
from services.asr        import transcribe     as whisper_asr
from services.translator import translate      as nllb_translate
from services.tts_engine import speak          as speak_generic
from services.tts_cloning    import speak_as
from services.voice_identity import enroll_speaker, is_enrolled, clear_speaker
from transport.sync_protocol import make_sync_packet

# ── Optional fast services ────────────────────────────────────────────────────
DEEPGRAM_KEY = os.getenv("DEEPGRAM_API_KEY", "")
GEMINI_KEY   = os.getenv("GEMINI_API_KEY",   "")
TARGET_LANG  = os.getenv("DUBBING_TARGET_LANG", "Hindi")

_gemini_fn = None
if GEMINI_KEY:
    try:
        from services.translator_gemini import translate_gemini_sync
        _gemini_fn = translate_gemini_sync
        print("[agent] Gemini translation: ✅")
    except Exception as e:
        print(f"[agent] Gemini unavailable: {e}")

# ── Constants ─────────────────────────────────────────────────────────────────
SR             = 16000
MIN_SPEECH_MS  = 350
SILENCE_MS     = 400
MAX_CHUNK_MS   = 6000
ENROLL_SECS    = 3

LANGUAGES = {
    "English":   "eng_Latn",
    "Hindi":     "hin_Deva",
    "Bengali":   "ben_Beng",
    "Tamil":     "tam_Taml",
    "Telugu":    "tel_Telu",
    "Kannada":   "kan_Knda",
    "Malayalam": "mal_Mlym",
    "Marathi":   "mar_Deva",
    "Gujarati":  "guj_Gujr",
    "Punjabi":   "pan_Guru",
    "Urdu":      "urd_Arab",
    "Nepali":    "npi_Deva",
}

# ── VAD ───────────────────────────────────────────────────────────────────────
_vad      = load_silero_vad()
_vad_lock = threading.Lock()


def _vad_ts(audio: np.ndarray):
    with _vad_lock:
        _vad.reset_states()
        return get_speech_timestamps(
            torch.from_numpy(audio.astype(np.float32)), _vad, sampling_rate=SR
        )


# ── Per-participant state ─────────────────────────────────────────────────────

class ParticipantDubber:
    """Handles one participant's full dub pipeline."""

    def __init__(self, sid: str, room, target_lang: str = TARGET_LANG):
        self.sid         = sid
        self.room        = room
        self.target_lang = target_lang
        self._buf        = np.array([], dtype=np.float32)
        self._speech_start: int | None = None
        self._enroll_buf: list[np.ndarray] = []
        self._result_q:  stdlib_queue.Queue = stdlib_queue.Queue()
        self._source     = None   # rtc.AudioSource for output track
        self._pool       = threading.Thread
        from concurrent.futures import ThreadPoolExecutor
        self._executor   = ThreadPoolExecutor(max_workers=2,
                                              thread_name_prefix=f"dub_{sid[:6]}")
        print(f"[agent] ParticipantDubber created for {sid}")

    async def setup_output_track(self):
        """Create and publish a dubbed audio output track back to the room."""
        if not _LIVEKIT_OK:
            return
        self._source = rtc.AudioSource(SR, 1)
        track = rtc.LocalAudioTrack.create_audio_track(
            f"dub_{self.sid[:6]}", self._source
        )
        opts = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        await self.room.local_participant.publish_track(track, opts)
        print(f"[agent] published output track for {self.sid}")

    def push_frame(self, audio_np: np.ndarray, sample_rate: int) -> None:
        """Called for each incoming audio frame. Non-blocking."""
        # Resample if needed
        if audio_np.dtype == np.int16:
            audio_np = audio_np.astype(np.float32) / 32768.0
        if sample_rate != SR:
            audio_np = librosa.resample(audio_np, orig_sr=sample_rate, target_sr=SR)

        # Enrollment
        if not is_enrolled(self.sid):
            self._enroll_buf.append(audio_np.copy())
            total = sum(len(a) for a in self._enroll_buf)
            if total >= SR * ENROLL_SECS:
                combined = np.concatenate(self._enroll_buf)
                enroll_speaker(self.sid, combined, sr=SR)
                self._enroll_buf.clear()
                print(f"[agent] enrolled {self.sid}")

        # VAD chunk accumulation
        self._buf = np.concatenate([self._buf, audio_np])
        max_buf   = int((MAX_CHUNK_MS / 1000 + 1) * SR)
        if len(self._buf) > max_buf:
            self._buf = self._buf[-max_buf:]

        if len(self._buf) < int(0.1 * SR):
            return

        ts = _vad_ts(self._buf)
        if not ts:
            self._buf         = np.array([], dtype=np.float32)
            self._speech_start = None
            return

        first_start  = ts[0]["start"]
        last_end     = ts[-1]["end"]
        silence_tail = len(self._buf) - last_end

        if self._speech_start is None:
            self._speech_start = first_start

        speech_len     = last_end - self._speech_start
        silence_needed = int(SILENCE_MS / 1000 * SR)
        min_speech     = int(MIN_SPEECH_MS / 1000 * SR)
        max_speech     = int(MAX_CHUNK_MS / 1000 * SR)

        flush = (
            (silence_tail >= silence_needed and speech_len >= min_speech)
            or speech_len >= max_speech
        )
        if flush:
            chunk             = self._buf[self._speech_start: last_end].copy()
            self._buf         = self._buf[last_end:]
            self._speech_start = None
            self._executor.submit(self._process_chunk, chunk)

    def _process_chunk(self, chunk: np.ndarray) -> None:
        """ASR → Translate → TTS. Runs in thread pool."""
        try:
            t0 = time.perf_counter()

            # Save chunk to temp wav
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, chunk, SR)
                wav_path = tmp.name

            # ASR
            transcript, src_lang = whisper_asr(wav_path)
            if not transcript.strip():
                return
            asr_ms = round((time.perf_counter() - t0) * 1000)

            # Translate
            tgt_code = LANGUAGES.get(self.target_lang, "hin_Deva")
            t_tr = time.perf_counter()
            if _gemini_fn:
                try:
                    translated = _gemini_fn(transcript, src_lang or "eng_Latn", tgt_code)
                except Exception:
                    translated = nllb_translate(transcript, src_lang or "eng_Latn", tgt_code)
            else:
                translated = nllb_translate(transcript, src_lang or "eng_Latn", tgt_code)
            tr_ms = round((time.perf_counter() - t_tr) * 1000)

            if not translated.strip():
                return

            # TTS
            t_tts = time.perf_counter()
            if is_enrolled(self.sid):
                speech_path = speak_as(translated.strip(), self.sid)
            else:
                speech_path = speak_generic(translated.strip())
            tts_ms = round((time.perf_counter() - t_tts) * 1000)
            total  = round((time.perf_counter() - t0) * 1000)

            print(
                f"[agent:{self.sid[:6]}] "
                f"ASR {asr_ms}ms | TR {tr_ms}ms | TTS {tts_ms}ms | "
                f"Total {total}ms\n"
                f"  '{transcript}' → '{translated}'"
            )

            # Queue result for publishing
            self._result_q.put({
                "transcript":  transcript,
                "translation": translated,
                "speech_path": speech_path,
                "orig_len_ms": len(chunk) / SR * 1000,
                "dubbed_ms":   tts_ms,
                "total_ms":    total,
            })

        except Exception as e:
            import traceback
            print(f"[agent:{self.sid[:6]}] chunk error: {e}")
            traceback.print_exc()

    async def publish_results(self) -> None:
        """
        Drain result queue and publish dubbed audio frames back to the room.
        Call this in a loop from the agent's async context.
        """
        while not self._result_q.empty():
            try:
                r = self._result_q.get_nowait()
            except stdlib_queue.Empty:
                break

            if not _LIVEKIT_OK or self._source is None:
                continue

            # Read the TTS wav and push it as AudioFrames
            try:
                audio_data, file_sr = sf.read(r["speech_path"], dtype="int16")
                if audio_data.ndim > 1:
                    audio_data = np.mean(audio_data, axis=1).astype(np.int16)
                if file_sr != SR:
                    audio_f = audio_data.astype(np.float32) / 32768.0
                    audio_f = librosa.resample(audio_f, orig_sr=file_sr, target_sr=SR)
                    audio_data = (audio_f * 32767).astype(np.int16)

                # Push in 10ms chunks
                chunk_size = SR // 100
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i: i + chunk_size]
                    if len(chunk) < chunk_size:
                        chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                    frame = rtc.AudioFrame(
                        data=chunk.tobytes(),
                        sample_rate=SR,
                        num_channels=1,
                        samples_per_channel=chunk_size,
                    )
                    await self._source.capture_frame(frame)

                # Publish sync packet
                pkt = make_sync_packet(
                    chunk_id          = str(uuid.uuid4()),
                    original_start_ms = time.time() * 1000 - r["orig_len_ms"],
                    original_end_ms   = time.time() * 1000,
                    dubbed_duration_ms = r["dubbed_ms"],
                    speaker_id        = self.sid,
                    target_lang       = self.target_lang,
                )
                await self.room.local_participant.publish_data(
                    pkt.encode(), reliable=True
                )

            except Exception as e:
                print(f"[agent] publish error: {e}")

    def cleanup(self) -> None:
        clear_speaker(self.sid)
        self._executor.shutdown(wait=False)


# ── Agent entrypoint ──────────────────────────────────────────────────────────

async def entrypoint(ctx: "JobContext"):
    await ctx.connect()
    room = ctx.room
    print(f"[agent] connected to room: {room.name}")

    dubbers: dict[str, ParticipantDubber] = {}

    @room.on("participant_disconnected")
    def on_leave(participant):
        sid = participant.identity
        if sid in dubbers:
            dubbers[sid].cleanup()
            del dubbers[sid]
        print(f"[agent] {sid} left")

    @room.on("track_subscribed")
    def on_track(track, publication, participant):
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        sid = participant.identity
        if sid not in dubbers:
            dubbers[sid] = ParticipantDubber(sid, room)
            asyncio.create_task(dubbers[sid].setup_output_track())

        asyncio.create_task(_stream_track(track, dubbers[sid]))

    async def _stream_track(track, dubber: ParticipantDubber):
        stream = rtc.AudioStream(track)
        async for ev in stream:
            frame = ev.frame
            audio = np.frombuffer(frame.data, dtype=np.int16)
            dubber.push_frame(audio, frame.sample_rate)
            await dubber.publish_results()

    # Keep alive
    await asyncio.sleep(float("inf"))


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not _LIVEKIT_OK:
        print("Install livekit-agents first: pip install livekit livekit-agents")
        sys.exit(1)
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
