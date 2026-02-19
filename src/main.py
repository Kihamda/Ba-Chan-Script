from __future__ import annotations

import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, cast

import flet as ft
import numpy as np
import sounddevice as sd

from app.flet_logic import AudioTranscriber
from app.audio_utils import (
    decode_audio_to_array,
    denoise_audio,
    encode_audio_to_file,
    trim_audio,
    resample_mono_float32,
)
from app.whisper_engine import EnginePool, ModelConfig

DEFAULT_TRANSCRIBE_SAMPLE_RATE = 16000
DEFAULT_CAPTURE_SAMPLE_RATE = 48000
DEFAULT_LANGUAGE = "ja"


def _format_srt_timestamp(seconds: float) -> str:
    total_ms = max(int(round(seconds * 1000.0)), 0)
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _build_srt(segments: list[tuple[float, float, str]]) -> str:
    lines: list[str] = []
    for idx, (start, end, text) in enumerate(segments, start=1):
        clean = text.strip()
        if not clean:
            continue
        lines.append(str(idx))
        lines.append(
            f"{_format_srt_timestamp(start)} --> {_format_srt_timestamp(end)}"
        )
        lines.append(clean)
        lines.append("")
    if not lines:
        return ""
    return "\n".join(lines).rstrip() + "\n"


def _compute_options(device: str) -> list[ft.dropdown.Option]:
    if device == "cuda":
        values = ["float16", "int8_float16", "int8"]
    elif device == "cpu":
        values = ["int8", "int8_float16"]
    else:
        values = ["float16", "int8_float16", "int8"]
    return [ft.dropdown.Option(v) for v in values]


def main(page: ft.Page) -> None:
    page.title = "速攻文字起こし (Flet + faster-whisper)"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.window_width = 900  # type: ignore[attr-defined]
    page.window_height = 900  # type: ignore[attr-defined]
    page.padding = 16
    page.horizontal_alignment = ft.CrossAxisAlignment.STRETCH

    engine_pool = EnginePool()
    transcriber: Optional[AudioTranscriber] = None
    file_thread: Optional[threading.Thread] = None

    recordings_dir = Path(__file__).resolve().parent.parent / "recordings"
    recordings_dir.mkdir(parents=True, exist_ok=True)
    last_recording_path: Optional[Path] = None
    last_recording_audio: Optional[np.ndarray] = None
    last_recording_sample_rate: int = DEFAULT_CAPTURE_SAMPLE_RATE

    edit_source_path: Optional[Path] = None
    edit_audio: Optional[np.ndarray] = None
    edit_sample_rate: int = DEFAULT_CAPTURE_SAMPLE_RATE
    last_transcript_text: str = ""
    last_transcript_segments: list[tuple[float, float, str]] = []
    preload_lock = threading.Lock()
    preloading_cfg: Optional[ModelConfig] = None
    loaded_cfg: Optional[ModelConfig] = None

    models_dir = Path(__file__).resolve().parent.parent / "models"

    model_dropdown = ft.Dropdown(
        label="モデル",
        options=[
            ft.dropdown.Option("base", "Base (高速)"),
            ft.dropdown.Option("small", "Small"),
            ft.dropdown.Option("medium", "Medium"),
            ft.dropdown.Option("large-v3", "Large v3 (高精度)"),
        ],
        value="base",
        dense=True,
    )

    device_dropdown = ft.Dropdown(
        label="デバイス",
        options=[
            ft.dropdown.Option("auto", "自動"),
            ft.dropdown.Option("cuda", "GPU"),
            ft.dropdown.Option("cpu", "CPU"),
        ],
        value="auto",
        dense=True,
    )

    compute_dropdown = ft.Dropdown(
        label="計算タイプ",
        options=_compute_options("auto"),
        value="float16",
        dense=True,
    )

    auto_preload_switch = ft.Switch(label="モデル事前ロード", value=True)
    preload_status = ft.Text(value="モデル状態: 未ロード", color=ft.Colors.GREY)
    preload_button = ft.Button(content="今すぐロード", icon=ft.Icons.DOWNLOAD)

    def refresh_compute_options(_: ft.ControlEvent) -> None:
        compute_dropdown.options = _compute_options(device_dropdown.value or "auto")
        compute_dropdown.value = compute_dropdown.options[0].key
        page.update()
        if auto_preload_switch.value:
            _preload_model_async(current_config())

    device_dropdown.on_change = refresh_compute_options
    model_dropdown.on_change = (
        lambda _: _preload_model_async(current_config())
        if auto_preload_switch.value
        else None
    )
    compute_dropdown.on_change = (
        lambda _: _preload_model_async(current_config())
        if auto_preload_switch.value
        else None
    )

    # ---- Recorder (Voice Memos-like) ----
    recorder_status = ft.Text("停止中", color=ft.Colors.GREY)
    recorder_level = ft.ProgressBar(
        value=0,
        bar_height=10,
        color=ft.Colors.RED,
        bgcolor=ft.Colors.GREY_200,
        border_radius=6,
    )
    recorder_transcript = ft.TextField(
        value="",
        label="リアルタイム文字起こし (直近30秒)",
        multiline=True,
        read_only=True,
        border=ft.InputBorder.OUTLINE,
        expand=True,
    )

    def _list_input_devices() -> list[ft.dropdown.Option]:
        try:
            devices = cast(list[dict[str, Any]], list(sd.query_devices()))
            hostapis = cast(list[Any], list(sd.query_hostapis()))

            def hostapi_name(device_dict: dict[str, Any]) -> str:
                try:
                    idx = int(device_dict.get("hostapi", -1))
                    if 0 <= idx < len(hostapis):
                        ha = hostapis[idx]
                        if isinstance(ha, dict):
                            return str(ha.get("name", ""))
                        # Fallback for unexpected types
                        return str(getattr(ha, "name", "") or "")
                except Exception:
                    pass
                return ""

            def rank_hostapi(name: str) -> int:
                n = (name or "").lower()
                if "wasapi" in n:
                    return 0
                if "wdm" in n or "kernel" in n or "ks" in n:
                    return 1
                if "asio" in n:
                    return 2
                if "directsound" in n:
                    return 3
                if "mme" in n:
                    return 4
                return 9

            def normalize_name(name: str) -> str:
                return " ".join((name or "").strip().lower().split())

            opts: list[ft.dropdown.Option] = [ft.dropdown.Option("default", "既定")]

            # De-duplicate by (normalized name). Pick the "best" host API.
            chosen: dict[str, tuple[int, dict]] = {}
            for idx, d in enumerate(devices):
                if int(d.get("max_input_channels", 0) or 0) <= 0:
                    continue
                name = str(d.get("name", f"Input {idx}"))
                key = normalize_name(name)
                ha = hostapi_name(d)
                score = (rank_hostapi(ha), -int(d.get("max_input_channels", 0) or 0))

                prev = chosen.get(key)
                if prev is None:
                    chosen[key] = (idx, d)
                    continue
                prev_idx, prev_d = prev
                prev_ha = hostapi_name(prev_d)
                prev_score = (
                    rank_hostapi(prev_ha),
                    -int(prev_d.get("max_input_channels", 0) or 0),
                )
                if score < prev_score:
                    chosen[key] = (idx, d)

            for _key in sorted(chosen.keys()):
                idx, d = chosen[_key]
                name = str(d.get("name", f"Input {idx}"))
                ha = hostapi_name(d)
                label = f"{name} [{ha}]" if ha else name
                opts.append(ft.dropdown.Option(str(idx), label))
            return opts
        except Exception:
            return [ft.dropdown.Option("default", "既定")]

    input_device_dropdown = ft.Dropdown(
        label="入力デバイス",
        options=_list_input_devices(),
        value="default",
        dense=True,
    )

    def refresh_input_devices(_: ft.ControlEvent) -> None:
        current = input_device_dropdown.value
        input_device_dropdown.options = _list_input_devices()
        keys = {o.key for o in input_device_dropdown.options}
        input_device_dropdown.value = current if current in keys else "default"
        page.update()

    refresh_devices_button = ft.Button(
        content="デバイス再読み込み",
        icon=ft.Icons.REFRESH,
        on_click=refresh_input_devices,
    )

    capture_sample_rate_dropdown = ft.Dropdown(
        label="録音サンプルレート",
        options=[
            ft.dropdown.Option("16000", "16k (軽い/低音質)"),
            ft.dropdown.Option("44100", "44.1k"),
            ft.dropdown.Option("48000", "48k (おすすめ)"),
        ],
        value=str(DEFAULT_CAPTURE_SAMPLE_RATE),
        dense=True,
    )

    latency_dropdown = ft.Dropdown(
        label="レイテンシ",
        options=[
            ft.dropdown.Option("default", "既定"),
            ft.dropdown.Option("low", "low"),
            ft.dropdown.Option("high", "high"),
        ],
        value="default",
        dense=True,
    )

    blocksize_dropdown = ft.Dropdown(
        label="ブロックサイズ",
        options=[
            ft.dropdown.Option("auto", "自動"),
            ft.dropdown.Option("256", "256"),
            ft.dropdown.Option("512", "512"),
            ft.dropdown.Option("1024", "1024"),
            ft.dropdown.Option("2048", "2048"),
        ],
        value="auto",
        dense=True,
    )

    file_status = ft.Text("待機中", color=ft.Colors.GREY)
    file_busy = ft.ProgressRing(visible=False, width=16, height=16)
    file_output = ft.TextField(
        value="",
        multiline=True,
        read_only=True,
        border=ft.InputBorder.OUTLINE,
        expand=True,
    )
    file_progress = ft.ProgressBar(value=0)

    edit_source_text = ft.Text(value="編集対象: なし")
    latest_recording_text = ft.Text(value="最新録音: なし")

    export_format = ft.Dropdown(
        label="形式",
        options=[
            ft.dropdown.Option("wav", "WAV"),
            ft.dropdown.Option("mp3", "MP3"),
            ft.dropdown.Option("flac", "FLAC"),
        ],
        value="wav",
        dense=True,
        width=120,
    )
    trim_range = ft.RangeSlider(
        min=0,
        max=30,
        start_value=0,
        end_value=30,
        divisions=300,
        label="{value}",
    )
    trim_start_text = ft.TextField(label="開始 mm:ss", value="00:00", width=140)
    trim_end_text = ft.TextField(label="終了 mm:ss", value="00:00", width=140)
    apply_trim_button = ft.Button(
        content="トリム適用",
        icon=ft.Icons.CONTENT_CUT,
        disabled=True,
    )
    denoise_button = ft.Button(
        content="ノイズ除去",
        icon=ft.Icons.AUTO_FIX_HIGH,
        disabled=True,
    )

    transcribe_edited_button = ft.Button(
        content="編集結果を文字起こし",
        icon=ft.Icons.AUDIO_FILE,
        disabled=True,
    )
    export_txt_button = ft.Button(
        content="TXT保存",
        icon=ft.Icons.DESCRIPTION,
        disabled=True,
    )
    export_srt_button = ft.Button(
        content="SRT保存",
        icon=ft.Icons.SUBTITLES,
        disabled=True,
    )
    save_edited_button = ft.Button(
        content="保存",
        icon=ft.Icons.SAVE,
        disabled=True,
    )
    open_source_button = ft.Button(
        content="音声ファイルを開く",
        icon=ft.Icons.FOLDER_OPEN,
    )
    use_latest_recording_button = ft.Button(
        content="最新録音を開く",
        icon=ft.Icons.HISTORY,
        disabled=True,
    )

    def on_preload_click(_: ft.ControlEvent) -> None:
        _preload_model_async(current_config())

    preload_button.on_click = on_preload_click

    def on_auto_preload_change(_: ft.ControlEvent) -> None:
        if auto_preload_switch.value:
            _preload_model_async(current_config())

    auto_preload_switch.on_change = on_auto_preload_change

    def current_config() -> ModelConfig:
        model_size = model_dropdown.value or "base"
        device = device_dropdown.value or "auto"
        compute_type = compute_dropdown.value
        batch_size = 8
        best_of = 5
        cpu_threads: int | None = None
        num_workers: int | None = None
        if model_size.startswith("large"):
            batch_size = 4
            best_of = 6

        cpu_count = os.cpu_count() or 4
        if device == "cpu":
            cpu_threads = max(cpu_count - 1, 1)
            num_workers = max(min(cpu_count // 2, 4), 1)

        return ModelConfig(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            cache_dir=models_dir,
            batch_size=batch_size,
            best_of=best_of,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
        )

    def _set_preload_status(text: str, color: str) -> None:
        preload_status.value = text
        preload_status.color = color
        page.update()

    def _preload_model_async(cfg: ModelConfig) -> None:
        nonlocal loaded_cfg
        nonlocal preloading_cfg

        with preload_lock:
            if loaded_cfg == cfg:
                _set_preload_status("モデル状態: ロード済み", ft.Colors.GREEN)
                return
            if preloading_cfg == cfg:
                return
            preloading_cfg = cfg

        _set_preload_status("モデル状態: ロード中...", ft.Colors.BLUE)

        def work() -> None:
            nonlocal loaded_cfg
            nonlocal preloading_cfg
            try:
                engine_pool.preload(cfg, warmup=True)
                loaded_cfg = cfg
                page.run_thread(
                    lambda: _set_preload_status(
                        "モデル状態: ロード済み", ft.Colors.GREEN
                    )
                )
            except Exception as exc:  # noqa: BLE001
                msg = f"モデル状態: ロード失敗 ({exc})"
                page.run_thread(
                    lambda m=msg: _set_preload_status(m, ft.Colors.RED)
                )
            finally:
                with preload_lock:
                    preloading_cfg = None

        threading.Thread(target=work, daemon=True).start()

    def realtime_update(text: str, is_final: bool) -> None:
        # 過去ログは持たない。常に「直近バッファの全文」を表示してローリング。
        def _apply() -> None:
            recorder_transcript.value = text.strip()
            page.update()

        page.run_thread(_apply)

    def on_level_update(level: float) -> None:
        def _apply() -> None:
            recorder_level.value = level
            page.update()

        page.run_thread(_apply)

    def _set_edit_audio(
        audio: np.ndarray,
        source_label: str,
        source_path: Optional[Path],
        sample_rate: int,
    ) -> None:
        nonlocal edit_audio
        nonlocal edit_source_path
        nonlocal edit_sample_rate

        edit_audio = audio
        edit_source_path = source_path
        edit_sample_rate = int(sample_rate)

        edit_source_text.value = f"編集対象: {source_label}"
        apply_trim_button.disabled = False
        denoise_button.disabled = False
        transcribe_edited_button.disabled = False
        save_edited_button.disabled = False

        # update trim UI defaults
        duration = audio.size / edit_sample_rate
        trim_range.max = max(duration, 1e-3)
        trim_range.start_value = 0
        trim_range.end_value = duration
        trim_start_text.value = "00:00"
        mm = int(duration) // 60
        ss = int(duration) % 60
        trim_end_text.value = f"{mm:02d}:{ss:02d}"

        # clear previous output
        file_output.value = ""
        file_progress.value = 0
        page.update()

    def stop_realtime() -> None:
        nonlocal transcriber
        nonlocal last_recording_path
        nonlocal last_recording_audio
        nonlocal last_recording_sample_rate
        if transcriber:
            transcriber.stop()

            audio = transcriber.get_recording_array()
            if audio.size > 0:
                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                out_path = recordings_dir / f"recording-{ts}.wav"

                capture_sr = transcriber.capture_sample_rate

                # Always keep a WAV for internal reference
                encode_audio_to_file(
                    samples=audio,
                    file_path=str(out_path),
                    sample_rate=capture_sr,
                    fmt="wav",
                )

                last_recording_path = out_path
                last_recording_audio = audio
                last_recording_sample_rate = capture_sr
                latest_recording_text.value = f"最新録音: {out_path.name}"
                use_latest_recording_button.disabled = False
                _set_edit_audio(audio, out_path.name, out_path, capture_sr)

                # auto switch to File tab
                tabs.selected_index = 1

            transcriber = None
        recorder_status.value = "停止中"
        recorder_status.color = ft.Colors.GREY
        recorder_level.value = 0
        record_button.content = "録音"
        record_button.icon = ft.Icons.MIC
        page.update()

    def start_realtime(_: ft.ControlEvent) -> None:
        nonlocal transcriber
        if transcriber:
            stop_realtime()
            return

        cfg = current_config()
        recorder_status.value = "録音中"
        recorder_status.color = ft.Colors.RED
        record_button.content = "停止"
        record_button.icon = ft.Icons.STOP
        page.update()

        device_value = input_device_dropdown.value
        selected_device: int | str | None
        if device_value and device_value != "default":
            try:
                selected_device = int(device_value)
            except ValueError:
                selected_device = device_value
        else:
            selected_device = None

        try:
            capture_sr = int(
                capture_sample_rate_dropdown.value or DEFAULT_CAPTURE_SAMPLE_RATE
            )
        except Exception:
            capture_sr = DEFAULT_CAPTURE_SAMPLE_RATE

        latency_value: str | float | None
        if (latency_dropdown.value or "default") == "default":
            latency_value = None
        else:
            latency_value = latency_dropdown.value

        blocksize_value: int | None
        if (blocksize_dropdown.value or "auto") == "auto":
            blocksize_value = None
        else:
            try:
                bs = blocksize_dropdown.value
                blocksize_value = int(bs) if bs is not None else None
            except Exception:
                blocksize_value = None

        transcriber = AudioTranscriber(
            on_text_update=realtime_update,
            on_level_update=on_level_update,
            engine_pool=engine_pool,
            model_config=cfg,
            capture_sample_rate=capture_sr,
            transcribe_sample_rate=DEFAULT_TRANSCRIBE_SAMPLE_RATE,
            device=selected_device,
            language=DEFAULT_LANGUAGE,
            fast_beam_size=1,
            final_beam_size=3,
            clear_on_final=False,
            latency=latency_value,
            blocksize=blocksize_value,
        )
        try:
            transcriber.start()
        except Exception as exc:  # noqa: BLE001
            stop_realtime()
            recorder_status.value = f"エラー: {exc}"
            recorder_status.color = ft.Colors.RED
            page.update()

    record_button = ft.Button(
        content="録音",
        icon=ft.Icons.MIC,
        on_click=start_realtime,
        style=ft.ButtonStyle(
            color=ft.Colors.WHITE,
            bgcolor=ft.Colors.RED,
            padding=18,
        ),
    )

    # FilePicker は service
    file_picker = ft.FilePicker()

    def use_latest_recording(_: ft.ControlEvent) -> None:
        nonlocal last_recording_path
        nonlocal last_recording_audio
        nonlocal last_recording_sample_rate
        if last_recording_audio is None:
            return
        label = last_recording_path.name if last_recording_path else "latest-recording"
        _set_edit_audio(
            last_recording_audio,
            label,
            last_recording_path,
            last_recording_sample_rate,
        )

    use_latest_recording_button.on_click = use_latest_recording

    async def open_source(_: ft.ControlEvent) -> None:
        files = await file_picker.pick_files(
            allow_multiple=False,
            file_type=ft.FilePickerFileType.AUDIO,
        )
        if not files or not files[0].path:
            return
        path = files[0].path
        file_status.value = "読み込み中"
        file_status.color = ft.Colors.BLUE
        page.update()

        def work() -> None:
            nonlocal edit_audio
            nonlocal edit_source_path
            nonlocal edit_sample_rate
            try:
                try:
                    target_sr = int(
                        capture_sample_rate_dropdown.value
                        or DEFAULT_CAPTURE_SAMPLE_RATE
                    )
                except Exception:
                    target_sr = DEFAULT_CAPTURE_SAMPLE_RATE

                audio = decode_audio_to_array(path, sample_rate=target_sr)
                page.run_thread(
                    lambda: _set_edit_audio(
                        audio, Path(path).name, Path(path), target_sr
                    )
                )
                page.run_thread(lambda: setattr(file_status, "value", "読み込み完了"))
                page.run_thread(lambda: setattr(file_status, "color", ft.Colors.GREEN))
            except Exception as exc:  # noqa: BLE001
                page.run_thread(
                    lambda: setattr(file_status, "value", f"読み込みエラー: {exc}")
                )
                page.run_thread(lambda: setattr(file_status, "color", ft.Colors.RED))
                page.run_thread(page.update)

        threading.Thread(target=work, daemon=True).start()

    open_source_button.on_click = open_source

    async def save_edited(_: ft.ControlEvent) -> None:
        nonlocal edit_audio
        nonlocal edit_source_path
        if edit_audio is None:
            return

        fmt = export_format.value or "wav"
        base = edit_source_path.stem if edit_source_path else "edited"
        dest = await file_picker.save_file(
            dialog_title="編集結果を保存",
            file_name=f"{base}.{fmt}",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=[fmt],
        )
        if not dest:
            return

        try:
            encode_audio_to_file(
                samples=edit_audio,
                file_path=dest,
                sample_rate=edit_sample_rate,
                fmt=fmt,  # type: ignore[arg-type]
            )
            file_status.value = "保存しました"
            file_status.color = ft.Colors.GREEN
        except Exception as exc:  # noqa: BLE001
            file_status.value = f"保存エラー: {exc}"
            file_status.color = ft.Colors.RED
        page.update()

    save_edited_button.on_click = save_edited

    async def export_txt(_: ft.ControlEvent) -> None:
        nonlocal last_transcript_text
        if not last_transcript_text.strip():
            return
        dest = await file_picker.save_file(
            dialog_title="文字起こし結果をTXT保存",
            file_name="transcript.txt",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["txt"],
        )
        if not dest:
            return
        try:
            Path(dest).write_text(last_transcript_text.strip() + "\n", encoding="utf-8")
            file_status.value = "TXTを保存しました"
            file_status.color = ft.Colors.GREEN
        except Exception as exc:  # noqa: BLE001
            file_status.value = f"TXT保存エラー: {exc}"
            file_status.color = ft.Colors.RED
        page.update()

    async def export_srt(_: ft.ControlEvent) -> None:
        nonlocal last_transcript_segments
        if not last_transcript_segments:
            return
        dest = await file_picker.save_file(
            dialog_title="文字起こし結果をSRT保存",
            file_name="transcript.srt",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["srt"],
        )
        if not dest:
            return
        try:
            content = _build_srt(last_transcript_segments)
            if not content:
                raise ValueError("有効な字幕セグメントがありません")
            Path(dest).write_text(content, encoding="utf-8")
            file_status.value = "SRTを保存しました"
            file_status.color = ft.Colors.GREEN
        except Exception as exc:  # noqa: BLE001
            file_status.value = f"SRT保存エラー: {exc}"
            file_status.color = ft.Colors.RED
        page.update()

    export_txt_button.on_click = export_txt
    export_srt_button.on_click = export_srt

    def _parse_mmss(value: str) -> Optional[int]:
        value = (value or "").strip()
        if not value:
            return None
        try:
            mm, ss = value.split(":")
            return int(mm) * 60 + int(ss)
        except Exception:
            return None

    def _apply_trim_from_ui() -> None:
        nonlocal edit_audio
        if edit_audio is None:
            return
        start_sec = float(trim_range.start_value)
        end_sec = float(trim_range.end_value)
        edit_audio = trim_audio(
            edit_audio,
            edit_sample_rate,
            start_sec,
            end_sec,
        )
        if edit_audio.size == 0:
            file_status.value = "トリム結果が空です"
            file_status.color = ft.Colors.RED
        else:
            file_status.value = "トリムしました"
            file_status.color = ft.Colors.GREEN

            duration = edit_audio.size / edit_sample_rate
            trim_range.max = max(duration, 1e-3)
            trim_range.start_value = 0
            trim_range.end_value = duration
            trim_start_text.value = "00:00"
            mm = int(duration) // 60
            ss = int(duration) % 60
            trim_end_text.value = f"{mm:02d}:{ss:02d}"

        page.update()

    def _sync_range_to_text(_: ft.ControlEvent) -> None:
        start = int(float(trim_range.start_value))
        end = int(float(trim_range.end_value))
        trim_start_text.value = f"{start // 60:02d}:{start % 60:02d}"
        trim_end_text.value = f"{end // 60:02d}:{end % 60:02d}"
        page.update()

    def _sync_text_to_range(_: ft.ControlEvent) -> None:
        start = _parse_mmss(trim_start_text.value)
        end = _parse_mmss(trim_end_text.value)
        if start is None or end is None:
            return
        if end <= start:
            end = start + 1
        trim_range.start_value = max(float(start), 0.0)
        trim_range.end_value = max(float(end), float(trim_range.start_value) + 1.0)
        page.update()

    trim_range.on_change = _sync_range_to_text
    trim_start_text.on_change = _sync_text_to_range
    trim_end_text.on_change = _sync_text_to_range

    def on_apply_trim(_: ft.ControlEvent) -> None:
        _apply_trim_from_ui()

    apply_trim_button.on_click = on_apply_trim

    def on_denoise(_: ft.ControlEvent) -> None:
        nonlocal edit_audio
        if edit_audio is None:
            return
        file_status.value = "ノイズ除去中"
        file_status.color = ft.Colors.BLUE
        page.update()

        def work() -> None:
            nonlocal edit_audio
            try:
                assert edit_audio is not None
                out = denoise_audio(edit_audio, edit_sample_rate)
                edit_audio = out
                page.run_thread(lambda: setattr(file_status, "value", "ノイズ除去完了"))
                page.run_thread(lambda: setattr(file_status, "color", ft.Colors.GREEN))
            except Exception as exc:  # noqa: BLE001
                msg = f"ノイズ除去エラー: {exc}"
                page.run_thread(lambda m=msg: setattr(file_status, "value", m))
                page.run_thread(lambda: setattr(file_status, "color", ft.Colors.RED))
            finally:
                page.run_thread(page.update)

        threading.Thread(target=work, daemon=True).start()

    denoise_button.on_click = on_denoise

    def run_edited_transcription(cfg: ModelConfig) -> None:
        nonlocal file_thread
        nonlocal edit_audio
        nonlocal last_transcript_text
        nonlocal last_transcript_segments

        def apply_status(text: str, color: str) -> None:
            def _():
                file_status.value = text
                file_status.color = color
                page.update()

            page.run_thread(_)

        if edit_audio is None:
            apply_status("編集対象がありません", ft.Colors.RED)
            return

        try:
            apply_status("解析中", ft.Colors.BLUE)
            page.run_thread(lambda: setattr(file_busy, "visible", True))
            engine = engine_pool.get(cfg)

            audio_for_stt = resample_mono_float32(
                edit_audio, edit_sample_rate, DEFAULT_TRANSCRIBE_SAMPLE_RATE
            )
            segments, info = engine.transcribe_ndarray(
                audio_for_stt,
                language=DEFAULT_LANGUAGE,
                beam_size=5 if cfg.model_size.startswith("large") else 3,
                silence_ms=500,
                chunk_length=30,
            )
            segment_list = list(segments)
            text = "".join(s.text for s in segment_list).strip()
            segment_payload = [
                (float(s.start), float(s.end), s.text)
                for s in segment_list
                if s.text.strip()
            ]

            def _done() -> None:
                nonlocal last_transcript_text
                nonlocal last_transcript_segments
                file_output.value = text
                last_transcript_text = text
                last_transcript_segments = segment_payload
                file_progress.value = 1
                file_status.value = f"完了 / 言語: {info.language}"
                file_status.color = ft.Colors.GREEN
                file_busy.visible = False
                transcribe_edited_button.disabled = False
                export_txt_button.disabled = not bool(text)
                export_srt_button.disabled = not bool(segment_payload)
                page.update()

            page.run_thread(_done)
        except Exception as exc:  # noqa: BLE001
            apply_status(f"エラー: {exc}", ft.Colors.RED)
            page.run_thread(lambda: setattr(file_busy, "visible", False))
            page.run_thread(
                lambda: setattr(transcribe_edited_button, "disabled", False)
            )
            page.run_thread(lambda: setattr(export_txt_button, "disabled", True))
            page.run_thread(lambda: setattr(export_srt_button, "disabled", True))
            page.run_thread(page.update)
        finally:
            file_thread = None

    def start_edited_transcription(_: ft.ControlEvent) -> None:
        nonlocal file_thread
        nonlocal last_transcript_text
        nonlocal last_transcript_segments
        if file_thread and file_thread.is_alive():
            return
        if edit_audio is None:
            file_status.value = "先に音声ファイルを開くか、録音してください"
            file_status.color = ft.Colors.RED
            page.update()
            return

        transcribe_edited_button.disabled = True
        file_busy.visible = True
        file_output.value = ""
        last_transcript_text = ""
        last_transcript_segments = []
        export_txt_button.disabled = True
        export_srt_button.disabled = True
        file_progress.value = 0
        file_status.value = "ジョブ投入"
        file_status.color = ft.Colors.BLUE
        page.update()

        cfg = current_config()
        file_thread = threading.Thread(
            target=run_edited_transcription, args=(cfg,), daemon=True
        )
        file_thread.start()

    transcribe_edited_button.on_click = start_edited_transcription

    recorder_settings = ft.Card(
        content=ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text("モデルとハードウェア設定", weight=ft.FontWeight.BOLD),
                    ft.ResponsiveRow(
                        controls=[
                            ft.Column(col=6, controls=[input_device_dropdown]),
                            ft.Column(col=6, controls=[model_dropdown]),
                            ft.Column(col=6, controls=[device_dropdown]),
                            ft.Column(col=6, controls=[compute_dropdown]),
                            ft.Column(col=6, controls=[refresh_devices_button]),
                            ft.Column(col=6, controls=[auto_preload_switch]),
                            ft.Column(col=6, controls=[preload_button]),
                            ft.Column(col=12, controls=[preload_status]),
                        ],
                    ),
                ],
                spacing=10,
            ),
            padding=16,
        )
    )

    recording_quality_settings = ft.Card(
        content=ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text("録音設定", weight=ft.FontWeight.BOLD),
                    ft.ResponsiveRow(
                        controls=[
                            ft.Column(col=6, controls=[capture_sample_rate_dropdown]),
                            ft.Column(col=6, controls=[latency_dropdown]),
                            ft.Column(col=6, controls=[blocksize_dropdown]),
                            ft.Column(
                                col=6,
                                controls=[
                                    ft.Text(
                                        "文字起こしは内部で16kHzへ変換します",
                                        color=ft.Colors.GREY,
                                    )
                                ],
                            ),
                        ],
                    ),
                ],
                spacing=10,
            ),
            padding=16,
        )
    )

    recorder_card = ft.Card(
        content=ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text("ボイスレコーダー", weight=ft.FontWeight.BOLD),
                    ft.Row(
                        controls=[
                            recorder_status,
                        ],
                    ),
                    recorder_level,
                    record_button,
                    recorder_transcript,
                ],
                spacing=10,
            ),
            padding=16,
        ),
        expand=True,
    )

    file_card = ft.Card(
        content=ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text("無制限ファイル文字起こし", weight=ft.FontWeight.BOLD),
                    latest_recording_text,
                    edit_source_text,
                    ft.Row(
                        controls=[
                            open_source_button,
                            use_latest_recording_button,
                            save_edited_button,
                            export_format,
                        ],
                        spacing=10,
                        wrap=True,
                    ),
                    ft.Text("編集", weight=ft.FontWeight.BOLD),
                    trim_range,
                    ft.Row(
                        controls=[
                            trim_start_text,
                            trim_end_text,
                            apply_trim_button,
                            denoise_button,
                        ],
                        spacing=10,
                        wrap=True,
                    ),
                    ft.ResponsiveRow(
                        controls=[
                            ft.Column(
                                col=4,
                                controls=[
                                    ft.Row(
                                        controls=[file_status, file_busy],
                                        spacing=8,
                                    ),
                                    transcribe_edited_button,
                                    ft.Row(
                                        controls=[
                                            export_txt_button,
                                            export_srt_button,
                                        ],
                                        spacing=8,
                                        wrap=True,
                                    ),
                                    file_progress,
                                ],
                                spacing=10,
                            ),
                            ft.Column(
                                col=8,
                                controls=[
                                    ft.Text(
                                        "文字起こし結果", weight=ft.FontWeight.W_600
                                    ),
                                    file_output,
                                ],
                                spacing=10,
                            ),
                        ],
                    ),
                ],
                spacing=10,
            ),
            padding=16,
        ),
        expand=True,
    )

    tab_bar = ft.TabBar(
        tabs=[
            ft.Tab(label="録音"),
            ft.Tab(label="ファイル"),
            ft.Tab(label="設定"),
        ]
    )
    tab_view = ft.TabBarView(
        controls=[
            ft.Column(
                controls=[
                    recorder_card,
                ],
                expand=True,
                spacing=10,
            ),
            file_card,
            ft.Column(
                controls=[
                    recorder_settings,
                    recording_quality_settings,
                ],
                expand=True,
                spacing=10,
            ),
        ],
        expand=1,
    )
    tabs = ft.Tabs(
        length=3,
        selected_index=0,
        content=ft.Column(
            controls=[
                tab_bar,
                tab_view,
            ],
            spacing=0,
            expand=1,
        ),
        expand=1,
    )

    root = ft.Column(
        controls=[
            tabs,
        ],
        expand=True,
        spacing=10,
    )

    page.add(root)

    if auto_preload_switch.value:
        _preload_model_async(current_config())

    page.on_disconnect = lambda _: stop_realtime()


if __name__ == "__main__":
    ft.run(main)
