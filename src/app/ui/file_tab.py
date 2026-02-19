from __future__ import annotations

import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import flet as ft
import numpy as np

from app.audio_utils import (
    decode_audio_to_array,
    denoise_audio,
    encode_audio_to_file,
    trim_audio,
    resample_mono_float32,
)
from app.whisper_engine import EnginePool, ModelConfig

DEFAULT_TRANSCRIBE_SAMPLE_RATE = 16000


class FileTab(ft.Container):
    def __init__(
        self,
        page: ft.Page,
        engine_pool: EnginePool,
        get_current_config: Callable[[], ModelConfig],
    ):
        super().__init__()
        self.page_ref = page
        self.engine_pool = engine_pool
        self.get_current_config = get_current_config

        # State
        self.edit_source_path: Optional[Path] = None
        self.edit_audio: Optional[np.ndarray] = None
        self.edit_sample_rate: int = 48000
        self.file_thread: Optional[threading.Thread] = None
        self.last_recording_audio: Optional[np.ndarray] = None
        self.last_recording_path: Optional[Path] = None
        self.last_recording_sr: int = 48000

        # UI Elements
        self.file_picker = ft.FilePicker()
        self.file_picker.on_result = self._on_file_upload
        self.save_picker = ft.FilePicker()
        self.save_picker.on_result = self._on_save_file
        self.page_ref.overlay.extend([self.file_picker, self.save_picker])

        self.file_status = ft.Text("待機中", color=ft.Colors.GREY)
        self.file_busy = ft.ProgressRing(visible=False, width=16, height=16)
        self.file_output = ft.TextField(
            value="",
            multiline=True,
            read_only=True,
            border=ft.InputBorder.OUTLINE,
            expand=True,
        )
        self.file_progress = ft.ProgressBar(value=0)

        self.edit_source_text = ft.Text(value="編集対象: なし")
        self.latest_recording_text = ft.Text(value="最新録音: なし")

        self.export_format = ft.Dropdown(
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

        self.trim_range = ft.RangeSlider(
            min=0,
            max=30,
            start_value=0,
            end_value=30,
            divisions=300,
            label="{value}",
            on_change=self._sync_range_to_text,
        )
        self.trim_start_text = ft.TextField(
            label="開始 mm:ss", value="00:00", width=140, on_change=self._sync_text_to_range
        )
        self.trim_end_text = ft.TextField(
            label="終了 mm:ss", value="00:00", width=140, on_change=self._sync_text_to_range
        )

        self.apply_trim_button = ft.Button(
            content="トリム適用",
            icon=ft.Icons.CONTENT_CUT,
            disabled=True,
            on_click=self._on_apply_trim,
        )
        self.denoise_button = ft.Button(
            content="ノイズ除去",
            icon=ft.Icons.AUTO_FIX_HIGH,
            disabled=True,
            on_click=self._on_denoise,
        )
        self.transcribe_edited_button = ft.Button(
            content="編集結果を文字起こし",
            icon=ft.Icons.AUDIO_FILE,
            disabled=True,
            on_click=self._start_edited_transcription,
        )
        self.save_edited_button = ft.Button(
            content="保存",
            icon=ft.Icons.SAVE,
            disabled=True,
            on_click=self._open_save_dialog,
        )
        self.open_source_button = ft.Button(
            content="音声ファイルを開く",
            icon=ft.Icons.FOLDER_OPEN,
            on_click=lambda _: self.file_picker.pick_files(
                allow_multiple=False, file_type=ft.FilePickerFileType.AUDIO
            ),
        )
        self.use_latest_recording_button = ft.Button(
            content="最新録音を開く",
            icon=ft.Icons.HISTORY,
            disabled=True,
            on_click=self._use_latest_recording,
        )

        # Layout
        self.controls = [
            ft.Card(
                content=ft.Container(
                    content=ft.Column(
                        controls=[
                            ft.Text("無制限ファイル文字起こし", weight=ft.FontWeight.BOLD),
                            self.latest_recording_text,
                            self.edit_source_text,
                            ft.Row(
                                controls=[
                                    self.open_source_button,
                                    self.use_latest_recording_button,
                                    self.save_edited_button,
                                    self.export_format,
                                ],
                                spacing=10,
                                wrap=True,
                            ),
                            ft.Text("編集", weight=ft.FontWeight.BOLD),
                            self.trim_range,
                            ft.Row(
                                controls=[
                                    self.trim_start_text,
                                    self.trim_end_text,
                                    self.apply_trim_button,
                                    self.denoise_button,
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
                                                controls=[self.file_status, self.file_busy],
                                                spacing=8,
                                            ),
                                            self.transcribe_edited_button,
                                            self.file_progress,
                                        ],
                                        spacing=10,
                                    ),
                                    ft.Column(
                                        col=8,
                                        controls=[
                                            ft.Text("文字起こし結果", weight=ft.FontWeight.W_600),
                                            self.file_output,
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
        ]
        self.expand = True

    def set_latest_recording(self, audio: np.ndarray, path: Path, sr: int) -> None:
        self.last_recording_audio = audio
        self.last_recording_path = path
        self.last_recording_sr = sr
        self.latest_recording_text.value = f"最新録音: {path.name}"
        self.use_latest_recording_button.disabled = False
        self.update()

        # Auto load
        self._set_edit_audio(audio, path.name, path, sr)

    def _use_latest_recording(self, _e: ft.ControlEvent) -> None:
        if self.last_recording_audio is not None:
            label = self.last_recording_path.name if self.last_recording_path else "latest"
            self._set_edit_audio(
                self.last_recording_audio,
                label,
                self.last_recording_path,
                self.last_recording_sr,
            )

    def _set_edit_audio(
        self,
        audio: np.ndarray,
        source_label: str,
        source_path: Optional[Path],
        sample_rate: int,
    ) -> None:
        self.edit_audio = audio
        self.edit_source_path = source_path
        self.edit_sample_rate = int(sample_rate)

        self.edit_source_text.value = f"編集対象: {source_label}"
        self.apply_trim_button.disabled = False
        self.denoise_button.disabled = False
        self.transcribe_edited_button.disabled = False
        self.save_edited_button.disabled = False

        # update trim UI defaults
        duration = audio.size / self.edit_sample_rate
        self.trim_range.max = max(duration, 1e-3)
        self.trim_range.start_value = 0
        self.trim_range.end_value = duration
        self.trim_start_text.value = "00:00"
        mm = int(duration) // 60
        ss = int(duration) % 60
        self.trim_end_text.value = f"{mm:02d}:{ss:02d}"

        # clear previous output
        self.file_output.value = ""
        self.file_progress.value = 0
        self.update()

    def _on_file_upload(self, e: ft.FilePickerResultEvent) -> None:
        if not e.files or not e.files[0].path:
            return
        path = e.files[0].path
        self.file_status.value = "読み込み中"
        self.file_status.color = ft.Colors.BLUE
        self.update()

        def work() -> None:
            try:
                # Always load at 48k or whatever default capture is, for simplicity
                target_sr = 48000
                audio = decode_audio_to_array(path, sample_rate=target_sr)
                
                def _done() -> None:
                    self._set_edit_audio(audio, Path(path).name, Path(path), target_sr)
                    self.file_status.value = "読み込み完了"
                    self.file_status.color = ft.Colors.GREEN
                    self.update()

                self.page_ref.run_thread(_done)
            except Exception as exc:
                def _err() -> None:
                    self.file_status.value = f"読み込みエラー: {exc}"
                    self.file_status.color = ft.Colors.RED
                    self.update()
                
                self.page_ref.run_thread(_err)

        threading.Thread(target=work, daemon=True).start()

    def _open_save_dialog(self, _e: ft.ControlEvent) -> None:
        fmt = self.export_format.value or "wav"
        base = self.edit_source_path.stem if self.edit_source_path else "edited"
        self.save_picker.save_file(
            dialog_title="編集結果を保存",
            file_name=f"{base}.{fmt}",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=[fmt],
        )

    def _on_save_file(self, e: ft.FilePickerResultEvent) -> None:
        dest = e.path
        if not dest or self.edit_audio is None:
            return

        try:
            fmt = self.export_format.value or "wav"
            encode_audio_to_file(
                samples=self.edit_audio,
                file_path=dest,
                sample_rate=self.edit_sample_rate,
                fmt=fmt,
            )
            self.file_status.value = "保存しました"
            self.file_status.color = ft.Colors.GREEN
        except Exception as exc:
            self.file_status.value = f"保存エラー: {exc}"
            self.file_status.color = ft.Colors.RED
        self.update()

    def _parse_mmss(self, value: str) -> Optional[int]:
        value = (value or "").strip()
        if not value:
            return None
        try:
            mm, ss = value.split(":")
            return int(mm) * 60 + int(ss)
        except Exception:
            return None

    def _on_apply_trim(self, _e: ft.ControlEvent) -> None:
        if self.edit_audio is None:
            return
        start_sec = float(self.trim_range.start_value)
        end_sec = float(self.trim_range.end_value)
        
        self.edit_audio = trim_audio(
            self.edit_audio,
            self.edit_sample_rate,
            start_sec,
            end_sec,
        )
        if self.edit_audio.size == 0:
            self.file_status.value = "トリム結果が空です"
            self.file_status.color = ft.Colors.RED
        else:
            self.file_status.value = "トリムしました"
            self.file_status.color = ft.Colors.GREEN
            
            # Reset timeline
            duration = self.edit_audio.size / self.edit_sample_rate
            self.trim_range.max = max(duration, 1e-3)
            self.trim_range.start_value = 0
            self.trim_range.end_value = duration
            self.trim_start_text.value = "00:00"
            mm = int(duration) // 60
            ss = int(duration) % 60
            self.trim_end_text.value = f"{mm:02d}:{ss:02d}"

        self.update()

    def _sync_range_to_text(self, _e: ft.ControlEvent) -> None:
        start = int(float(self.trim_range.start_value))
        end = int(float(self.trim_range.end_value))
        self.trim_start_text.value = f"{start // 60:02d}:{start % 60:02d}"
        self.trim_end_text.value = f"{end // 60:02d}:{end % 60:02d}"
        self.update()

    def _sync_text_to_range(self, _e: ft.ControlEvent) -> None:
        start = self._parse_mmss(self.trim_start_text.value)
        end = self._parse_mmss(self.trim_end_text.value)
        if start is None or end is None:
            return
        if end <= start:
            end = start + 1
        self.trim_range.start_value = max(float(start), 0.0)
        self.trim_range.end_value = max(float(end), float(self.trim_range.start_value) + 1.0)
        self.update()

    def _on_denoise(self, _e: ft.ControlEvent) -> None:
        if self.edit_audio is None:
            return
        self.file_status.value = "ノイズ除去中"
        self.file_status.color = ft.Colors.BLUE
        self.update()

        def work() -> None:
            try:
                out = denoise_audio(self.edit_audio, self.edit_sample_rate)
                
                def _done() -> None:
                    self.edit_audio = out
                    self.file_status.value = "ノイズ除去完了"
                    self.file_status.color = ft.Colors.GREEN
                    self.update()
                
                self.page_ref.run_thread(_done)
            except Exception as exc:
                def _err() -> None:
                    self.file_status.value = f"ノイズ除去エラー: {exc}"
                    self.file_status.color = ft.Colors.RED
                    self.update()
                
                self.page_ref.run_thread(_err)

        threading.Thread(target=work, daemon=True).start()

    def _start_edited_transcription(self, _e: ft.ControlEvent) -> None:
        if self.file_thread and self.file_thread.is_alive():
            return
        if self.edit_audio is None:
            self.file_status.value = "先に音声ファイルを開くか、録音してください"
            self.file_status.color = ft.Colors.RED
            self.update()
            return

        self.transcribe_edited_button.disabled = True
        self.file_busy.visible = True
        self.file_output.value = ""
        self.file_progress.value = 0
        self.file_status.value = "ジョブ投入"
        self.file_status.color = ft.Colors.BLUE
        self.update()

        cfg = self.get_current_config()
        self.file_thread = threading.Thread(
            target=self._run_edited_transcription, args=(cfg,), daemon=True
        )
        self.file_thread.start()

    def _run_edited_transcription(self, cfg: ModelConfig) -> None:
        try:
            def _set_status(msg: str, color: str) -> None:
                def _() -> None:
                    self.file_status.value = msg
                    self.file_status.color = color
                    self.update()
                self.page_ref.run_thread(_)

            _set_status("解析中", ft.Colors.BLUE)
            
            engine = self.engine_pool.get(cfg)

            audio_for_stt = resample_mono_float32(
                self.edit_audio, self.edit_sample_rate, DEFAULT_TRANSCRIBE_SAMPLE_RATE
            )
            segments, info = engine.transcribe_ndarray(
                audio_for_stt,
                language="ja",
                beam_size=5 if cfg.model_size.startswith("large") else 3,
                silence_ms=500,
                chunk_length=30,
            )
            text = "".join(s.text for s in segments).strip()

            def _done() -> None:
                self.file_output.value = text
                self.file_progress.value = 1
                self.file_status.value = f"完了 / 言語: {info.language}"
                self.file_status.color = ft.Colors.GREEN
                self.file_busy.visible = False
                self.transcribe_edited_button.disabled = False
                self.update()

            self.page_ref.run_thread(_done)
        except Exception as exc:
            def _err() -> None:
                self.file_status.value = f"エラー: {exc}"
                self.file_status.color = ft.Colors.RED
                self.file_busy.visible = False
                self.transcribe_edited_button.disabled = False
                self.update()
            
            self.page_ref.run_thread(_err)
        finally:
            self.file_thread = None
