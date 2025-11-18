from __future__ import annotations

import base64
import os
import time
import uuid
from dataclasses import dataclass, field

import numpy as np
from flask import Blueprint, current_app, jsonify, render_template, request
from flask_socketio import emit
from werkzeug.utils import secure_filename

from app import socketio
from app.audio_utils import decode_audio_bytes, pcm16le_to_array
from app.whisper_engine import get_fast_engine, get_precise_engine

bp = Blueprint("main", __name__)

ALLOWED_EXTENSIONS = {"mp3", "wav", "mp4", "mpeg", "mpga", "m4a", "webm", "flac", "ogg"}


@dataclass
class RealtimeSessionState:
    sample_rate: int
    chunks: list[np.ndarray] = field(default_factory=list)
    total_samples: int = 0
    info_sent: bool = False
    offset_seconds: float = 0.0
    history_chunks: list[np.ndarray] = field(default_factory=list)
    history_samples: int = 0
    archive_chunks: list[np.ndarray] = field(default_factory=list)
    last_emit_end: float = 0.0
    last_snapshot_at: float = 0.0
    last_precise_emit: float = 0.0
    precise_task_running: bool = False
    cleanup_pending: bool = False
    full_run_started: bool = False
    full_task_running: bool = False
    timeline_index: dict[tuple[int, int], dict] = field(default_factory=dict)

    def append(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        self.chunks.append(samples)
        self.total_samples += samples.size
        history_chunk = samples.copy()
        self.history_chunks.append(history_chunk)
        self.history_samples += history_chunk.size
        self.archive_chunks.append(history_chunk.copy())
        self._shrink_buffer()
        self._trim_history()

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

    def history_audio(self, window_seconds: float) -> tuple[np.ndarray, float]:
        if not self.sample_rate or not self.history_chunks:
            return np.empty(0, dtype=np.float32), self.offset_seconds

        needed_samples = int(self.sample_rate * window_seconds)
        needed_samples = max(needed_samples, self.total_samples)

        collected: list[np.ndarray] = []
        count = 0
        for chunk in reversed(self.history_chunks):
            collected.append(chunk)
            count += chunk.size
            if count >= needed_samples:
                break

        if not collected:
            return np.empty(0, dtype=np.float32), self.offset_seconds

        audio = np.concatenate(list(reversed(collected)))
        if count > needed_samples:
            audio = audio[-needed_samples:]

        start_offset = max(0.0, self.total_elapsed() - audio.size / self.sample_rate)
        return audio, start_offset

    def full_audio(self) -> np.ndarray:
        if not self.archive_chunks:
            return np.empty(0, dtype=np.float32)
        if len(self.archive_chunks) == 1:
            return self.archive_chunks[0]
        return np.concatenate(self.archive_chunks)


_realtime_sessions: dict[str, RealtimeSessionState] = {}
MIN_BUFFER_SECONDS = 0.3
SNAPSHOT_REFRESH_SECONDS = 2.0
ROLLING_WINDOW_SECONDS = 30.0
MAX_BUFFER_SECONDS = 8.0
MAX_HISTORY_SECONDS = 300.0
PRECISION_WINDOW_SECONDS = 90.0
PRECISION_REFRESH_SECONDS = 15.0
FAST_BEAM_SIZE = 3
TIMESTAMP_BUCKETS_PER_SECOND = 20  # 50ms resolution
TIMELINE_RETENTION_PAD = 2.0


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/api/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "ファイルがありません"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "ファイルが選択されていません"}), 400

    if not allowed_file(file.filename or ""):
        return jsonify({"error": "対応していないファイル形式です"}), 400

    language = request.form.get("language", "ja")
    beam_size = int(request.form.get("beam_size", 5))

    filename = secure_filename(file.filename or "upload")
    unique_filename = f"{uuid.uuid4()}_{filename}"
    filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], unique_filename)
    file.save(filepath)

    try:
        engine = get_precise_engine()
        start_time = time.time()
        result = engine.transcribe_file(
            filepath, language=language, beam_size=beam_size
        )
        elapsed = time.time() - start_time
        result["processing_time"] = elapsed
        result["filename"] = filename
        return jsonify(result)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@socketio.on("connect")
def handle_connect():
    print("Client connected")
    emit("status", {"message": "接続しました"})


@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")


@socketio.on("start_realtime_transcribe")
def handle_realtime_transcribe(data):
    sid = getattr(request, "sid", None)
    try:
        audio_data = data.get("audio")
        language = data.get("language", "ja")
        mime_type = data.get("mimeType", "audio/webm")
        if not audio_data:
            emit("error", {"message": "音声データがありません"}, to=sid)
            return

        audio_bytes = base64.b64decode(
            audio_data.split(",")[1] if "," in audio_data else audio_data
        )
        _run_realtime_transcribe(
            audio_bytes,
            language,
            mime_type,
            session_id=str(uuid.uuid4()),
            chunk_index=0,
            is_final=True,
            sid=sid,
        )
    except Exception as exc:  # noqa: BLE001
        current_app.logger.exception("Realtime transcription failed")
        emit("error", {"message": str(exc)}, to=sid)


@socketio.on("realtime_chunk")
def handle_realtime_chunk(data):
    sid = getattr(request, "sid", None)
    try:
        audio_data = data.get("audio")
        language = data.get("language", "ja")
        session_id = data.get("sessionId") or str(uuid.uuid4())
        chunk_index = int(data.get("chunkIndex", 0))
        sample_rate = int(data.get("sampleRate") or 16000)
        is_final = bool(data.get("isFinal", False))

        session = _realtime_sessions.get(session_id)
        if session is None:
            session = RealtimeSessionState(sample_rate=sample_rate)
            session.last_snapshot_at = -SNAPSHOT_REFRESH_SECONDS
            _realtime_sessions[session_id] = session

        if not audio_data:
            if is_final:
                _transcribe_session(
                    session_id, session, language, chunk_index, True, sid
                )
                _realtime_sessions.pop(session_id, None)
            else:
                emit("error", {"message": "音声データがありません"}, to=sid)
            return

        audio_bytes = base64.b64decode(
            audio_data.split(",")[1] if "," in audio_data else audio_data
        )
        samples = pcm16le_to_array(audio_bytes, sample_rate)
        if samples.size == 0 and not is_final:
            return

        if session.sample_rate != sample_rate:
            samples = _resample_audio(samples, sample_rate, session.sample_rate)

        session.append(samples)
        current_app.logger.info(
            "Realtime chunk ingest session=%s chunk=%s duration=%.2fs total=%.2fs",
            session_id,
            chunk_index,
            samples.size / session.sample_rate if session.sample_rate else 0,
            session.total_elapsed(),
        )

        has_buffer = session.buffered_duration() >= MIN_BUFFER_SECONDS
        should_process = has_buffer and (
            session.total_elapsed() - session.last_snapshot_at
            >= SNAPSHOT_REFRESH_SECONDS
            or is_final
        )

        if should_process:
            _transcribe_session(
                session_id, session, language, chunk_index, is_final, sid
            )
            _maybe_schedule_precise(
                session_id, session, language, chunk_index, sid, force=is_final
            )

            if is_final:
                session.cleanup_pending = True
                _start_full_transcription(
                    session_id, session, language, chunk_index, sid
                )
                if not session.precise_task_running:
                    _realtime_sessions.pop(session_id, None)
    except Exception as exc:  # noqa: BLE001
        current_app.logger.exception("Realtime chunk transcription failed")
        emit("error", {"message": str(exc)}, to=sid)


def _run_realtime_transcribe(
    audio_bytes: bytes,
    language: str,
    mime_type: str,
    session_id: str,
    chunk_index: int,
    is_final: bool,
    sid: str | None = None,
) -> None:
    current_app.logger.info(
        "Realtime audio chunk session=%s idx=%s mime=%s size=%s bytes",
        session_id,
        chunk_index,
        mime_type,
        len(audio_bytes),
    )

    audio_array = decode_audio_bytes(audio_bytes)
    engine = get_fast_engine()
    for segment in engine.transcribe_stream(
        audio_array, language=language, beam_size=FAST_BEAM_SIZE
    ):
        payload = dict(segment)
        payload.update({"sessionId": session_id, "chunkIndex": chunk_index})
        emit("transcribe_segment", payload, to=sid)

    if is_final:
        emit(
            "transcribe_complete",
            {"message": "文字起こし完了", "sessionId": session_id},
            to=sid,
        )


def _transcribe_session(
    session_id: str,
    session: RealtimeSessionState,
    language: str,
    chunk_index: int,
    is_final: bool,
    sid: str | None = None,
) -> None:
    audio_array, offset = session.history_audio(ROLLING_WINDOW_SECONDS)
    if audio_array.size == 0:
        if is_final:
            emit(
                "transcribe_complete",
                {"message": "文字起こし完了", "sessionId": session_id},
                to=sid,
            )
        return

    engine = get_fast_engine()
    segments: list[dict] = []

    for item in engine.transcribe_stream(
        audio_array, language=language, beam_size=FAST_BEAM_SIZE
    ):
        if item.get("type") == "info":
            if not session.info_sent:
                payload = dict(item)
                payload.update({"sessionId": session_id, "chunkIndex": chunk_index})
                emit("transcribe_segment", payload, to=sid)
                session.info_sent = True
            continue

        segments.append(
            {
                "start": item["start"] + offset,
                "end": item["end"] + offset,
                "text": item["text"],
            }
        )

    segments = _deduplicate_segments(segments)
    timeline_segments = _merge_timeline(session, segments)

    window_start = timeline_segments[0]["start"] if timeline_segments else offset
    window_end = timeline_segments[-1]["end"] if timeline_segments else offset

    snapshot_payload = {
        "sessionId": session_id,
        "chunkIndex": chunk_index,
        "segments": timeline_segments,
        "windowStart": window_start,
        "windowEnd": window_end,
    }
    emit("transcribe_snapshot", snapshot_payload, to=sid)

    if timeline_segments:
        session.last_emit_end = timeline_segments[-1]["end"]
    session.last_snapshot_at = session.total_elapsed()

    if is_final:
        emit(
            "transcribe_complete",
            {"message": "文字起こし完了", "sessionId": session_id},
            to=sid,
        )
    else:
        _retain_tail_context(session)


def _retain_tail_context(
    session: RealtimeSessionState, keep_seconds: float = 1.0
) -> None:
    if not session.sample_rate:
        return
    keep_samples = int(session.sample_rate * keep_seconds)
    if session.total_samples <= keep_samples:
        return

    audio = session.audio_array()
    tail = audio[-keep_samples:]
    session.chunks = [tail]
    trimmed = session.total_samples - tail.size
    session.total_samples = tail.size
    session.offset_seconds += trimmed / session.sample_rate


def _maybe_schedule_precise(
    session_id: str,
    session: RealtimeSessionState,
    language: str,
    chunk_index: int,
    sid: str | None,
    force: bool = False,
) -> bool:
    if session.precise_task_running or not session.sample_rate:
        return False

    elapsed_since_precise = session.total_elapsed() - session.last_precise_emit
    if not force and elapsed_since_precise < PRECISION_REFRESH_SECONDS:
        return False

    audio_array, offset = session.history_audio(PRECISION_WINDOW_SECONDS)
    if audio_array.size == 0:
        return False

    session.precise_task_running = True
    socketio.start_background_task(
        _run_precise_window,
        session_id,
        language,
        chunk_index,
        sid,
        audio_array.copy(),
        offset,
        is_full=False,
    )
    return True


def _start_full_transcription(
    session_id: str,
    session: RealtimeSessionState,
    language: str,
    chunk_index: int,
    sid: str | None,
) -> None:
    if session.full_run_started:
        return

    audio_array = session.full_audio()
    if audio_array.size == 0:
        return

    session.full_run_started = True
    session.full_task_running = True
    socketio.start_background_task(
        _run_precise_window,
        session_id,
        language,
        chunk_index,
        sid,
        audio_array.copy(),
        0.0,
        is_full=True,
    )


def _run_precise_window(
    session_id: str,
    language: str,
    chunk_index: int,
    sid: str | None,
    audio_array: np.ndarray,
    offset_seconds: float,
    is_full: bool,
) -> None:
    try:
        engine = get_precise_engine()
        segments: list[dict] = []
        for item in engine.transcribe_stream(
            audio_array, language=language, beam_size=3
        ):
            if item.get("type") != "segment":
                continue
            segments.append(
                {
                    "start": item["start"] + offset_seconds,
                    "end": item["end"] + offset_seconds,
                    "text": item["text"],
                }
            )

        segments = _deduplicate_segments(segments)
        socketio.emit(
            "transcribe_precise",
            {
                "sessionId": session_id,
                "chunkIndex": chunk_index,
                "segments": segments,
                "windowStart": segments[0]["start"] if segments else offset_seconds,
                "windowEnd": segments[-1]["end"] if segments else offset_seconds,
                "isFull": is_full,
            },
            to=sid,
        )
    finally:
        session = _realtime_sessions.get(session_id)
        if session:
            if is_full:
                session.full_task_running = False
            else:
                session.precise_task_running = False
                session.last_precise_emit = session.total_elapsed()
            if (
                session.cleanup_pending
                and not session.precise_task_running
                and not session.full_task_running
            ):
                _realtime_sessions.pop(session_id, None)


def _resample_audio(
    samples: np.ndarray, source_rate: int, target_rate: int
) -> np.ndarray:
    if samples.size == 0 or source_rate == target_rate:
        return samples

    duration = samples.size / float(source_rate)
    target_size = max(1, int(duration * target_rate))
    x_old = np.linspace(0.0, samples.size - 1, num=samples.size, dtype=np.float32)
    x_new = np.linspace(0.0, samples.size - 1, num=target_size, dtype=np.float32)
    return np.interp(x_new, x_old, samples).astype(np.float32)


def _deduplicate_segments(segments: list[dict]) -> list[dict]:
    seen: set[tuple[int, int]] = set()
    unique: list[dict] = []
    for segment in segments:
        key = _segment_key(segment.get("start", 0.0), segment.get("end", 0.0))
        if key in seen:
            continue
        seen.add(key)
        unique.append(segment)
    return unique


def _segment_key(start: float, end: float) -> tuple[int, int]:
    bucket = TIMESTAMP_BUCKETS_PER_SECOND
    return (int(round(start * bucket)), int(round(end * bucket)))


def _merge_timeline(session: RealtimeSessionState, incoming: list[dict]) -> list[dict]:
    for segment in incoming:
        key = _segment_key(segment.get("start", 0.0), segment.get("end", 0.0))
        session.timeline_index[key] = segment
    return sorted(session.timeline_index.values(), key=lambda item: item["start"])


@bp.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})
