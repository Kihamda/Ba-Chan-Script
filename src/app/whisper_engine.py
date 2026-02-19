from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Iterable, Iterator, Optional, Tuple, Union

import numpy as np
from faster_whisper import BatchedInferencePipeline, WhisperModel
from faster_whisper.transcribe import Segment, TranscriptionInfo

AudioSource = Union[str, np.ndarray]
TranscribeResult = Tuple[Iterable[Segment], TranscriptionInfo]


@dataclass(frozen=True)
class ModelConfig:
    model_size: str
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    compute_type: str | None = None  # e.g. float16, int8_float16, int8
    cache_dir: Path = Path("./models")
    batch_size: int = 8
    best_of: int = 5
    temperature: tuple[float, ...] | float = (0.0, 0.2, 0.4)
    no_speech_threshold: float | None = 0.35
    language_detection_threshold: float | None = 0.7
    cpu_threads: int | None = None
    num_workers: int | None = None


def _resolve_device(device_pref: str, compute_type_pref: str | None) -> tuple[str, str]:
    """Resolve device/compute_type honoring env overrides and availability."""
    env_device = os.getenv("WHISPER_DEVICE")
    env_compute = os.getenv("WHISPER_COMPUTE_TYPE")
    if env_device:
        device_pref = env_device
    if env_compute:
        compute_type_pref = env_compute

    device_pref = device_pref.lower()
    device = "cuda" if device_pref in {"cuda", "gpu"} else device_pref

    if device == "auto":
        try:
            import ctranslate2

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"

    if compute_type_pref:
        return device, compute_type_pref

    if device == "cuda":
        return device, "float16"
    return device, "int8"


class WhisperEngine:
    """Whisper wrapper with lazy loading and shared pipelines."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.device, self.compute_type = _resolve_device(
            config.device, config.compute_type
        )
        self.model: Optional[WhisperModel] = None
        self.pipeline: Optional[BatchedInferencePipeline] = None
        self.lock = Lock()
        self.logger = logging.getLogger(
            f"WhisperEngine[{config.model_size}:{self.device}:{self.compute_type}]"
        )

    def _ensure_model(self) -> WhisperModel:
        if self.model is None:
            with self.lock:
                if self.model is None:
                    self.logger.info(
                        "Loading model=%s device=%s compute=%s",
                        self.config.model_size,
                        self.device,
                        self.compute_type,
                    )
                    cache_dir = Path(self.config.cache_dir)
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    model_kwargs: dict[str, object] = {
                        "device": self.device,
                        "compute_type": self.compute_type,
                        "download_root": str(cache_dir),
                    }
                    if self.config.cpu_threads is not None:
                        model_kwargs["cpu_threads"] = self.config.cpu_threads
                    if self.config.num_workers is not None:
                        model_kwargs["num_workers"] = self.config.num_workers

                    self.model = WhisperModel(self.config.model_size, **model_kwargs)
                    self.pipeline = BatchedInferencePipeline(model=self.model)
                    self.logger.info("Model ready")
        assert self.model is not None
        assert self.pipeline is not None
        return self.model

    def _transcribe(
        self,
        audio_source: AudioSource,
        language: str,
        beam_size: int,
        silence_ms: int,
        chunk_length: int = 30,
    ) -> TranscribeResult:
        self._ensure_model()
        assert self.pipeline is not None
        return self.pipeline.transcribe(
            audio_source,
            language=language,
            beam_size=beam_size,
            best_of=max(beam_size, self.config.best_of),
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": silence_ms},
            batch_size=self.config.batch_size,
            temperature=self.config.temperature,
            no_speech_threshold=self.config.no_speech_threshold,
            language_detection_threshold=self.config.language_detection_threshold,
            condition_on_previous_text=True,
            compression_ratio_threshold=2.2,
            chunk_length=chunk_length,
        )

    def warmup(self, sample_rate: int = 16000) -> None:
        """Load model and run a tiny silent inference to pre-initialize runtime kernels."""
        self._ensure_model()
        silent = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
        segments, _ = self.transcribe_ndarray(
            silent,
            language="ja",
            beam_size=1,
            silence_ms=200,
            chunk_length=5,
        )
        # Consume generator to force execution.
        for _ in segments:
            pass

    def transcribe_ndarray(
        self,
        audio_source: np.ndarray,
        language: str = "ja",
        beam_size: int = 1,
        silence_ms: int = 300,
        chunk_length: int = 15,
    ) -> TranscribeResult:
        return self._transcribe(
            audio_source,
            language=language,
            beam_size=beam_size,
            silence_ms=silence_ms,
            chunk_length=chunk_length,
        )

    def transcribe_file_stream(
        self,
        audio_path: str,
        language: str = "ja",
        beam_size: int = 5,
        silence_ms: int = 500,
        chunk_length: int = 30,
    ) -> Iterator[dict]:
        segments, info = self._transcribe(
            audio_path,
            language=language,
            beam_size=beam_size,
            silence_ms=silence_ms,
            chunk_length=chunk_length,
        )

        yield {
            "type": "info",
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
        }

        duration = info.duration or 0.0
        for segment in segments:
            progress = min(segment.end / duration, 1.0) if duration else 0.0
            yield {
                "type": "segment",
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "progress": progress,
            }


class EnginePool:
    """Cache WhisperEngine instances by config to avoid reloads."""

    def __init__(self):
        self._engines: dict[ModelConfig, WhisperEngine] = {}
        self._lock = Lock()

    def get(self, config: ModelConfig) -> WhisperEngine:
        with self._lock:
            engine = self._engines.get(config)
            if engine is None:
                engine = WhisperEngine(config)
                self._engines[config] = engine
        return engine

    def preload(self, config: ModelConfig, warmup: bool = True) -> WhisperEngine:
        engine = self.get(config)
        if warmup:
            engine.warmup()
        else:
            engine._ensure_model()
        return engine
