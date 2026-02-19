import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from app.whisper_engine import EnginePool, ModelConfig, WhisperEngine
from app.audio_utils import resample_mono_float32

# Keep logging minimal but available for diagnostics
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tunables for realtime behavior
MIN_BUFFER_SECONDS = 0.3
SNAPSHOT_REFRESH_SECONDS = 1.0
MAX_BUFFER_SECONDS = 30.0
MAX_HISTORY_SECONDS = 300.0
VAD_THRESHOLD = 0.01  # Simple RMS-based VAD threshold
MIN_SILENCE_FOR_FINAL = 0.7  # Seconds of silence to emit a final result
LEVEL_UPDATE_INTERVAL_SECONDS = 0.1


@dataclass
class RealtimeSessionState:
    sample_rate: int
    chunks: list[np.ndarray] = field(default_factory=list)
    total_samples: int = 0
    offset_seconds: float = 0.0
    history_chunks: list[np.ndarray] = field(default_factory=list)
    history_samples: int = 0
    last_snapshot_at: float = 0.0

    def append(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        self.chunks.append(samples)
        self.total_samples += samples.size
        history_chunk = samples.copy()
        self.history_chunks.append(history_chunk)
        self.history_samples += history_chunk.size
        self._shrink_buffer()
        self._trim_history()

    def clear_buffer(self) -> None:
        self.chunks = []
        self.total_samples = 0

    def buffered_duration(self) -> float:
        if not self.sample_rate:
            return 0.0
        return self.total_samples / self.sample_rate

    def total_elapsed(self) -> float:
        return self.offset_seconds + self.buffered_duration()

    def _shrink_buffer(self) -> None:
        if not self.sample_rate:
            return
        max_samples = int(self.sample_rate * MAX_BUFFER_SECONDS)
        while self.total_samples > max_samples and self.chunks:
            removed = self.chunks.pop(0)
            self.total_samples -= removed.size
            self.offset_seconds += removed.size / self.sample_rate

    def _trim_history(self) -> None:
        if not self.sample_rate:
            return
        max_history = int(self.sample_rate * MAX_HISTORY_SECONDS)
        while self.history_samples > max_history and self.history_chunks:
            removed = self.history_chunks.pop(0)
            self.history_samples -= removed.size

    def audio_array(self) -> np.ndarray:
        if not self.chunks:
            return np.empty(0, dtype=np.float32)
        if len(self.chunks) == 1:
            return self.chunks[0]
        return np.concatenate(self.chunks)


class AudioTranscriber:
    """Real-time microphone transcriber for Flet UI."""

    def __init__(
        self,
        on_text_update: Callable[[str, bool], None],
        engine_pool: EnginePool,
        model_config: ModelConfig,
        capture_sample_rate: int = 48000,
        transcribe_sample_rate: int = 16000,
        device: Optional[int | str] = None,
        fast_beam_size: int = 1,
        final_beam_size: int = 3,
        language: str = "ja",
        clear_on_final: bool = True,
        on_level_update: Optional[Callable[[float], None]] = None,
        latency: str | float | None = None,
        blocksize: int | None = None,
    ):
        self.on_text_update = on_text_update
        self.on_level_update = on_level_update
        self.engine_pool = engine_pool
        self.model_config = model_config
        self.capture_sample_rate = int(capture_sample_rate)
        self.transcribe_sample_rate = int(transcribe_sample_rate)
        self.device = device
        self.fast_beam_size = fast_beam_size
        self.final_beam_size = final_beam_size
        self.language = language
        self.clear_on_final = clear_on_final
        self.latency = latency
        self.blocksize = blocksize
        self.running = False
        self.stream: Optional[sd.InputStream] = None
        self.session = RealtimeSessionState(sample_rate=self.capture_sample_rate)
        self.lock = threading.Lock()
        self._recording_chunks: list[np.ndarray] = []
        self.processing_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # VAD state
        self.silence_duration = 0.0
        self.is_speech = False

        # Meter throttling
        self._last_level_emit_at = 0.0

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.stop_event.clear()
        self.session = RealtimeSessionState(sample_rate=self.capture_sample_rate)
        self._recording_chunks = []
        self.session.last_snapshot_at = -SNAPSHOT_REFRESH_SECONDS
        self.silence_duration = 0.0
        self.is_speech = False

        self.processing_thread = threading.Thread(
            target=self._process_loop, daemon=True
        )
        self.processing_thread.start()

        def _rate_candidates() -> list[int]:
            candidates: list[int] = []
            if self.capture_sample_rate > 0:
                candidates.append(self.capture_sample_rate)

            try:
                dev = sd.query_devices(self.device, kind="input")
                default_sr = int(float(dev.get("default_samplerate", 0) or 0))
                if default_sr > 0:
                    candidates.append(default_sr)
            except Exception:
                pass

            # Common fallbacks
            candidates.extend([48000, 44100, 16000])

            # De-dup while preserving order
            seen: set[int] = set()
            out: list[int] = []
            for r in candidates:
                if r not in seen and r > 0:
                    seen.add(r)
                    out.append(r)
            return out

        last_exc: Exception | None = None
        for rate in _rate_candidates():
            try:
                self.capture_sample_rate = int(rate)
                with self.lock:
                    self.session = RealtimeSessionState(
                        sample_rate=self.capture_sample_rate
                    )
                    self._recording_chunks = []
                    self.session.last_snapshot_at = -SNAPSHOT_REFRESH_SECONDS

                self.stream = sd.InputStream(
                    samplerate=self.capture_sample_rate,
                    channels=1,
                    dtype="float32",
                    callback=self._audio_callback,
                    device=self.device,
                    latency=self.latency,
                    blocksize=self.blocksize,
                )
                self.stream.start()
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                self.stream = None
                continue

        if last_exc is not None:
            logger.error("Failed to start audio stream: %s", last_exc)
            self.running = False
            self.stop_event.set()
            raise RuntimeError(f"入力ストリームを開始できませんでした: {last_exc}") from last_exc

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        self.stop_event.set()
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)

    def get_recording_array(self) -> np.ndarray:
        """Returns full recorded audio since start() as float32 mono array."""
        with self.lock:
            if not self._recording_chunks:
                return np.empty(0, dtype=np.float32)
            if len(self._recording_chunks) == 1:
                return self._recording_chunks[0].copy()
            return np.concatenate([c for c in self._recording_chunks]).copy()

    def _audio_callback(
        self, indata, frames, time_info, status
    ) -> None:  # noqa: ANN001
        if status:
            logger.warning("Audio status: %s", status)
        if not self.running:
            return

        samples = indata.flatten()
        rms = float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0
        duration = (
            samples.size / self.capture_sample_rate if self.capture_sample_rate else 0.0
        )

        # Peak meter (0..1) update, throttled
        if self.on_level_update is not None:
            now = time.monotonic()
            if now - self._last_level_emit_at >= LEVEL_UPDATE_INTERVAL_SECONDS:
                # Heuristic scaling: speech around 0.02-0.1 RMS
                level = min(max(rms * 20.0, 0.0), 1.0)
                self._last_level_emit_at = now
                try:
                    self.on_level_update(level)
                except Exception:
                    # Meter should never break audio callback
                    pass

        with self.lock:
            self.session.append(samples)
            self._recording_chunks.append(samples.copy())
            if rms > VAD_THRESHOLD:
                self.silence_duration = 0.0
                self.is_speech = True
            else:
                self.silence_duration += duration

    def _process_loop(self) -> None:
        while not self.stop_event.is_set():
            time.sleep(0.1)

            is_final = False
            should_process = False
            audio_data: Optional[np.ndarray] = None

            with self.lock:
                if self.is_speech and self.silence_duration > MIN_SILENCE_FOR_FINAL:
                    is_final = True
                    self.is_speech = False

                has_buffer = self.session.buffered_duration() >= MIN_BUFFER_SECONDS
                time_since_snapshot = (
                    self.session.total_elapsed() - self.session.last_snapshot_at
                )

                if has_buffer and (
                    time_since_snapshot >= SNAPSHOT_REFRESH_SECONDS or is_final
                ):
                    should_process = True
                    audio_data = self.session.audio_array()
                    self.session.last_snapshot_at = self.session.total_elapsed()

            if not (should_process and audio_data is not None and audio_data.size > 0):
                continue

            try:
                engine: WhisperEngine = self.engine_pool.get(self.model_config)
                beam_size = self.final_beam_size if is_final else self.fast_beam_size
                silence_ms = 500 if is_final else 300
                chunk_length = 30 if is_final else 15

                if self.capture_sample_rate != self.transcribe_sample_rate:
                    audio_for_stt = resample_mono_float32(
                        audio_data,
                        self.capture_sample_rate,
                        self.transcribe_sample_rate,
                    )
                else:
                    audio_for_stt = audio_data

                segments, _info = engine.transcribe_ndarray(
                    audio_for_stt,
                    language=self.language,
                    beam_size=beam_size,
                    silence_ms=silence_ms,
                    chunk_length=chunk_length,
                )
                text = "".join(segment.text for segment in segments)
                if text.strip():
                    self.on_text_update(text, is_final)

                if is_final and self.clear_on_final:
                    with self.lock:
                        self.session.clear_buffer()
            except Exception as exc:  # noqa: BLE001
                logger.error("Transcription failed: %s", exc)
