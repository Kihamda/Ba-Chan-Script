from __future__ import annotations

from faster_whisper import WhisperModel, BatchedInferencePipeline
from faster_whisper.transcribe import TranscriptionInfo, Segment
import os
from threading import Lock
from typing import Iterable, Iterator, Optional, Union, Tuple
import logging

import numpy as np

AudioSource = Union[str, np.ndarray]
TranscribeResult = Tuple[Iterable[Segment], TranscriptionInfo]


class WhisperEngine:
    """Faster-Whisperエンジンのラッパークラス"""

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        batch_size: int = 8,
        best_of: int = 5,
        temperature: tuple[float, ...] | float = (0.0, 0.2, 0.4),
        no_speech_threshold: float | None = 0.35,
        language_detection_threshold: float | None = 0.7,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.best_of = best_of
        self.temperature = temperature
        self.no_speech_threshold = no_speech_threshold
        self.language_detection_threshold = language_detection_threshold
        self.model: Optional[WhisperModel] = None
        self.pipeline: Optional[BatchedInferencePipeline] = None
        self.lock = Lock()
        self.logger = logging.getLogger(f"WhisperEngine[{model_size}]")

    def load_model(self) -> WhisperModel:
        """モデルをロード（遅延ロード）"""
        if self.model is None:
            with self.lock:
                if self.model is None:
                    self.logger.info(
                        "Loading Whisper model size=%s device=%s compute_type=%s",
                        self.model_size,
                        self.device,
                        self.compute_type,
                    )
                    self.model = WhisperModel(
                        self.model_size,
                        device=self.device,
                        compute_type=self.compute_type,
                        download_root="./models",
                    )
                    self.pipeline = BatchedInferencePipeline(model=self.model)
                    self.logger.info("Model loaded successfully")
        assert self.model is not None
        assert self.pipeline is not None
        return self.model

    def _transcribe(
        self, audio_source: AudioSource, language: str, beam_size: int, silence_ms: int
    ) -> TranscribeResult:
        self.load_model()
        assert self.pipeline is not None
        return self.pipeline.transcribe(
            audio_source,
            language=language,
            beam_size=beam_size,
            best_of=max(beam_size, self.best_of),
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=silence_ms),
            batch_size=self.batch_size,
            temperature=self.temperature,
            no_speech_threshold=self.no_speech_threshold,
            language_detection_threshold=self.language_detection_threshold,
            condition_on_previous_text=True,
            compression_ratio_threshold=2.2,
        )

    def transcribe_file(
        self,
        audio_path: str,
        language: str = "ja",
        beam_size: int = 7,
    ) -> dict:
        """
        ファイルから高精度文字起こし

        Args:
            audio_path: 音声ファイルパス
            language: 言語コード
            beam_size: ビームサイズ（大きいほど精度向上、遅くなる）

        Returns:
            文字起こし結果
        """
        segments, info = self._transcribe(
            audio_path, language, beam_size, silence_ms=500
        )

        result = {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "segments": [],
        }

        for segment in segments:
            result["segments"].append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "words": (
                        [
                            {
                                "start": word.start,
                                "end": word.end,
                                "word": word.word,
                                "probability": word.probability,
                            }
                            for word in segment.words
                        ]
                        if segment.words
                        else []
                    ),
                }
            )

        return result

    def transcribe_stream(
        self, audio_source: AudioSource, language: str = "ja", beam_size: int = 1
    ) -> Iterator[dict]:
        """
        リアルタイム用の高速文字起こし（ストリーミング）

        Args:
            audio_path: 音声ファイルパス
            language: 言語コード
            beam_size: ビームサイズ（リアルタイムは1推奨）

        Yields:
            セグメント単位の文字起こし結果
        """
        segments, info = self._transcribe(
            audio_source, language, beam_size, silence_ms=300
        )

        yield {
            "type": "info",
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
        }

        for segment in segments:
            yield {
                "type": "segment",
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
            }


# グローバルエンジンインスタンス
_fast_engine = None
_precise_engine = None


def _detect_device() -> tuple[str, str]:
    """Prefer CUDA if CTranslate2 can see it, otherwise fall back to CPU."""
    env_device = os.getenv("WHISPER_DEVICE")
    env_compute = os.getenv("WHISPER_COMPUTE_TYPE")
    if env_device and env_compute:
        return env_device, env_compute

    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass

    return "cpu", "int8"


def get_fast_engine() -> WhisperEngine:
    """リアルタイム用の高速エンジン（small or base）"""
    global _fast_engine
    if _fast_engine is None:
        device, compute_type = _detect_device()
        _fast_engine = WhisperEngine(
            model_size="base",  # リアルタイム用
            device=device,
            compute_type=compute_type,
            batch_size=6,
            best_of=4,
            temperature=(0.0, 0.2),
            no_speech_threshold=0.25,
        )
    return _fast_engine


def get_precise_engine() -> WhisperEngine:
    """高精度用のエンジン（large-v3）"""
    global _precise_engine
    if _precise_engine is None:
        device, compute_type = _detect_device()
        _precise_engine = WhisperEngine(
            model_size="large-v3",  # 高精度モデル
            device=device,
            compute_type=compute_type,
            batch_size=12,
            best_of=6,
            temperature=(0.0, 0.2, 0.4),
            no_speech_threshold=0.3,
        )
    return _precise_engine
