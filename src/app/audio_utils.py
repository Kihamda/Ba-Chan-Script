"""Audio helper utilities."""

from __future__ import annotations

from typing import Any, Literal, Union
import io

import av
from av.audio.resampler import AudioResampler
import numpy as np

DEFAULT_SAMPLE_RATE = 16000


def resample_mono_float32(
    samples: np.ndarray, src_rate: int, dst_rate: int
) -> np.ndarray:
    """Resample 1D mono float32 samples using PyAV.

    Expects samples normalized to [-1, 1]. Returns float32 mono.
    """
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError("サンプルレートが不正です")
    if samples.size == 0:
        return np.empty(0, dtype=np.float32)
    if src_rate == dst_rate:
        return samples.astype(np.float32, copy=False)

    mono = np.ascontiguousarray(samples.astype(np.float32, copy=False).reshape(1, -1))
    frame = av.AudioFrame.from_ndarray(mono, format="flt", layout="mono")
    frame.sample_rate = int(src_rate)

    resampler = AudioResampler(format="flt", layout="mono", rate=int(dst_rate))
    resampled = resampler.resample(frame)

    frames = resampled if isinstance(resampled, list) else [resampled]
    chunks: list[np.ndarray] = []
    for rf in frames:
        arr = rf.to_ndarray()
        chunks.append(arr)

    if not chunks:
        return np.empty(0, dtype=np.float32)
    out = np.concatenate(chunks, axis=1).flatten().astype(np.float32, copy=False)
    return out


def encode_audio_to_file(
    samples: np.ndarray,
    file_path: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    fmt: Literal["wav", "flac", "mp3"] = "wav",
) -> None:
    """Encode float32 mono samples [-1,1] to an audio file.

    Note: MP3 support depends on underlying FFmpeg/codec availability in PyAV.
    """
    if samples.size == 0:
        raise ValueError("音声データが空です")

    fmt = fmt.lower()  # type: ignore[assignment]
    if fmt not in {"wav", "flac", "mp3"}:
        raise ValueError("未対応の保存形式です")

    # Convert to PCM16 for encoding
    pcm = np.clip(samples.astype(np.float32), -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16)
    mono = pcm16.reshape(1, -1)

    if fmt == "wav":
        container_format = "wav"
        codec_name = "pcm_s16le"
    elif fmt == "flac":
        container_format = "flac"
        codec_name = "flac"
    else:
        container_format = "mp3"
        codec_name = "libmp3lame"

    try:
        with av.open(file_path, mode="w", format=container_format) as out:
            stream = out.add_stream(codec_name, rate=sample_rate)
            stream.layout = "mono"

            frame_size = 1024
            total = mono.shape[1]
            for i in range(0, total, frame_size):
                chunk = mono[:, i : i + frame_size]
                frame = av.AudioFrame.from_ndarray(chunk, format="s16", layout="mono")
                frame.sample_rate = sample_rate
                for packet in stream.encode(frame):
                    out.mux(packet)

            for packet in stream.encode(None):
                out.mux(packet)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"音声の保存に失敗しました: {e}") from e


def trim_audio(
    samples: np.ndarray,
    sample_rate: int,
    start_sec: float,
    end_sec: float,
) -> np.ndarray:
    if samples.size == 0:
        return samples
    start = max(int(start_sec * sample_rate), 0)
    end = min(int(end_sec * sample_rate), int(samples.size))
    if end <= start:
        return np.empty(0, dtype=np.float32)
    return samples[start:end].copy()


def denoise_audio(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """One-click denoise. Uses noisereduce (spectral gating)."""
    if samples.size == 0:
        return samples
    try:
        import noisereduce as nr

        reduced = nr.reduce_noise(y=samples.astype(np.float32), sr=sample_rate)
        return reduced.astype(np.float32)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"ノイズ除去に失敗しました: {e}") from e


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
