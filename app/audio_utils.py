"""Audio helper utilities."""

from __future__ import annotations

from typing import Any, Union
import io

import av
from av.audio.resampler import AudioResampler
import numpy as np

DEFAULT_SAMPLE_RATE = 16000


def decode_audio_to_array(
    file_path: str, sample_rate: int = DEFAULT_SAMPLE_RATE
) -> np.ndarray:
    """Decode arbitrary media file into mono float32 samples."""
    return _decode_container(av.open(file_path), sample_rate)


def decode_audio_bytes(
    data: Union[bytes, bytearray, memoryview], sample_rate: int = DEFAULT_SAMPLE_RATE
) -> np.ndarray:
    """Decode in-memory audio bytes into mono float32 samples."""
    if not data:
        raise ValueError("音声データが空です")
    buffer = io.BytesIO(data)
    buffer.seek(0)
    return _decode_container(av.open(buffer), sample_rate)


def _decode_container(container: Any, sample_rate: int) -> np.ndarray:
    try:
        stream = container.streams.audio[0]
        resampler = AudioResampler(
            format="s16",
            layout="mono",
            rate=sample_rate,
        )

        chunks: list[np.ndarray] = []

        for frame in container.decode(stream):
            resampled = resampler.resample(frame)

            if isinstance(resampled, list):
                frames = resampled
            else:
                frames = [resampled]

            for rf in frames:
                arr = rf.to_ndarray()
                chunks.append(arr)

        if not chunks:
            raise ValueError("音声データを解析できませんでした")

        audio = np.concatenate(chunks, axis=1).flatten().astype(np.float32) / 32768.0
        return audio

    except Exception as e:
        raise ValueError(f"音声データの読み込みに失敗しました: {e}") from e

    finally:
        container.close()


def pcm16le_to_array(
    data: Union[bytes, bytearray, memoryview], sample_rate: int = DEFAULT_SAMPLE_RATE
) -> np.ndarray:
    """Convert raw PCM16LE mono bytes to float32 numpy array normalized to [-1, 1]."""
    if not data:
        return np.empty(0, dtype=np.float32)

    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    audio /= 32768.0
    return audio
