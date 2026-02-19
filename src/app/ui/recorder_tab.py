from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import flet as ft
import numpy as np

from app.flet_logic import AudioTranscriber
from app.whisper_engine import EnginePool, ModelConfig
from app.audio_utils import encode_audio_to_file

# Callbacks
OnRecordingFinished = Callable[[np.ndarray, Path, int], None]


class RecorderTab(ft.Container):
    def __init__(
        self,
        page: ft.Page,
        engine_pool: EnginePool,
        recordings_dir: Path,
        get_current_config: Callable[[], ModelConfig],
        get_audio_settings: Callable[[], dict[str, Any]],
        on_recording_finished: OnRecordingFinished | None = None,
    ):
        super().__init__()
        self.page_ref = page
        self.engine_pool = engine_pool
        self.recordings_dir = recordings_dir
        self.get_current_config = get_current_config
        self.get_audio_settings = get_audio_settings
        self.on_recording_finished = on_recording_finished

        # State
        self.transcriber: Optional[AudioTranscriber] = None
        
        # UI Elements
        self.recorder_status = ft.Text("停止中", color=ft.Colors.GREY)
        self.recorder_level = ft.ProgressBar(
            value=0,
            bar_height=10,
            color=ft.Colors.RED,
            bgcolor=ft.Colors.GREY_200,
            border_radius=6,
        )
        self.recorder_transcript = ft.TextField(
            value="",
            label="リアルタイム文字起こし (直近30秒)",
            multiline=True,
            read_only=True,
            border=ft.InputBorder.OUTLINE,
            expand=True,
        )

        self.record_button = ft.Button(
            content="録音",
            icon=ft.Icons.MIC,
            on_click=self._toggle_recording,
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.RED,
                padding=18,
            ),
        )

        # Layout
        self.content = ft.Card(
            content=ft.Container(
                content=ft.Column(
                    controls=[
                        ft.Text("ボイスレコーダー", weight=ft.FontWeight.BOLD),
                        ft.Row(
                            controls=[
                                self.recorder_status,
                            ],
                        ),
                        self.recorder_level,
                        self.record_button,
                        self.recorder_transcript,
                    ],
                    spacing=10,
                ),
                padding=16,
            ),
            expand=True,
        )
        self.expand = True

    def _realtime_update(self, text: str, is_final: bool) -> None:
        def _apply() -> None:
            self.recorder_transcript.value = text.strip()
            self.recorder_transcript.update()

        self.page_ref.run_thread(_apply)

    def _on_level_update(self, level: float) -> None:
        def _apply() -> None:
            self.recorder_level.value = level
            self.recorder_level.update()

        self.page_ref.run_thread(_apply)

    def _stop_realtime(self) -> None:
        if self.transcriber:
            self.transcriber.stop()

            audio = self.transcriber.get_recording_array()
            if audio.size > 0:
                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                out_path = self.recordings_dir / f"recording-{ts}.wav"
                capture_sr = self.transcriber.capture_sample_rate

                try:
                    encode_audio_to_file(
                        samples=audio,
                        file_path=str(out_path),
                        sample_rate=capture_sr,
                        fmt="wav",
                    )
                except Exception as e:
                    logging.error(f"Failed to save recording: {e}")

                if self.on_recording_finished and out_path.exists():
                    self.on_recording_finished(audio, out_path, capture_sr)
            
            self.transcriber = None

        self.recorder_status.value = "停止中"
        self.recorder_status.color = ft.Colors.GREY
        self.recorder_level.value = 0
        self.record_button.content = "録音"
        self.record_button.icon = ft.Icons.MIC
        self.update()

    def _toggle_recording(self, _e: ft.ControlEvent) -> None:
        if self.transcriber:
            self._stop_realtime()
            return

        settings = self.get_audio_settings()
        cfg = self.get_current_config()

        self.recorder_status.value = "録音中"
        self.recorder_status.color = ft.Colors.RED
        self.record_button.content = "停止"
        self.record_button.icon = ft.Icons.STOP
        self.update()

        self.transcriber = AudioTranscriber(
            on_text_update=self._realtime_update,
            on_level_update=self._on_level_update,
            engine_pool=self.engine_pool,
            model_config=cfg,
            capture_sample_rate=settings.get("capture_sample_rate", 48000),
            transcribe_sample_rate=16000,
            device=settings.get("device"),
            language="ja",
            fast_beam_size=1,
            final_beam_size=3,
            clear_on_final=False,
            latency=settings.get("latency"),
            blocksize=settings.get("blocksize"),
        )

        try:
            self.transcriber.start()
        except Exception as exc:
            self._stop_realtime()
            self.recorder_status.value = f"エラー: {exc}"
            self.recorder_status.color = ft.Colors.RED
            self.update()

    def stop_if_running(self) -> None:
        if self.transcriber:
            self._stop_realtime()
