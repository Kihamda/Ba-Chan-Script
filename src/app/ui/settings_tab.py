from __future__ import annotations

from typing import Any, Callable, cast

import flet as ft
import sounddevice as sd

from app.whisper_engine import ModelConfig

DEFAULT_CAPTURE_SAMPLE_RATE = 48000


def _compute_options(device: str) -> list[ft.dropdown.Option]:
    if device == "cuda":
        values = ["float16", "int8_float16", "int8"]
    elif device == "cpu":
        values = ["int8", "int8_float16"]
    else:
        values = ["float16", "int8_float16", "int8"]
    return [ft.dropdown.Option(v) for v in values]


class SettingsTab(ft.Column):
    def __init__(self, page: ft.Page, on_config_changed: Callable[[], None] | None = None):
        super().__init__()
        self.page_ref = page
        self.on_config_changed = on_config_changed
        self.expand = True
        self.spacing = 10

        # Model & Hardware Settings
        self.model_dropdown = ft.Dropdown(
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
        self.model_dropdown.on_change = self._notify_config_change

        self.device_dropdown = ft.Dropdown(
            label="デバイス",
            options=[
                ft.dropdown.Option("auto", "自動"),
                ft.dropdown.Option("cuda", "GPU"),
                ft.dropdown.Option("cpu", "CPU"),
            ],
            value="auto",
            dense=True,
        )
        self.device_dropdown.on_change = self._on_device_change

        self.compute_dropdown = ft.Dropdown(
            label="計算タイプ",
            options=_compute_options("auto"),
            value="float16",
            dense=True,
        )
        self.compute_dropdown.on_change = self._notify_config_change

        self.input_device_dropdown = ft.Dropdown(
            label="入力デバイス",
            options=self._list_input_devices(),
            value="default",
            dense=True,
        )
        self.input_device_dropdown.on_change = self._notify_config_change

        self.refresh_devices_button = ft.Button(
            content="デバイス再読み込み",
            icon=ft.Icons.REFRESH,
            on_click=self._refresh_input_devices,
        )

        # Recording Settings
        self.capture_sample_rate_dropdown = ft.Dropdown(
            label="録音サンプルレート",
            options=[
                ft.dropdown.Option("16000", "16k (軽い/低音質)"),
                ft.dropdown.Option("44100", "44.1k"),
                ft.dropdown.Option("48000", "48k (おすすめ)"),
            ],
            value=str(DEFAULT_CAPTURE_SAMPLE_RATE),
            dense=True,
        )
        self.capture_sample_rate_dropdown.on_change = self._notify_config_change

        self.latency_dropdown = ft.Dropdown(
            label="レイテンシ",
            options=[
                ft.dropdown.Option("default", "既定"),
                ft.dropdown.Option("low", "low"),
                ft.dropdown.Option("high", "high"),
            ],
            value="default",
            dense=True,
        )
        self.latency_dropdown.on_change = self._notify_config_change

        self.blocksize_dropdown = ft.Dropdown(
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
        self.blocksize_dropdown.on_change = self._notify_config_change

        # Layout
        self.controls = [
            ft.Card(
                content=ft.Container(
                    content=ft.Column(
                        controls=[
                            ft.Text("モデルとハードウェア設定", weight=ft.FontWeight.BOLD),
                            ft.ResponsiveRow(
                                controls=[
                                    ft.Column(col=6, controls=[self.input_device_dropdown]),
                                    ft.Column(col=6, controls=[self.model_dropdown]),
                                    ft.Column(col=6, controls=[self.device_dropdown]),
                                    ft.Column(col=6, controls=[self.compute_dropdown]),
                                    ft.Column(col=6, controls=[self.refresh_devices_button]),
                                ],
                            ),
                        ],
                        spacing=10,
                    ),
                    padding=16,
                )
            ),
            ft.Card(
                content=ft.Container(
                    content=ft.Column(
                        controls=[
                            ft.Text("録音設定", weight=ft.FontWeight.BOLD),
                            ft.ResponsiveRow(
                                controls=[
                                    ft.Column(col=6, controls=[self.capture_sample_rate_dropdown]),
                                    ft.Column(col=6, controls=[self.latency_dropdown]),
                                    ft.Column(col=6, controls=[self.blocksize_dropdown]),
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
            ),
        ]

    def _notify_config_change(self, _e: ft.ControlEvent) -> None:
        if self.on_config_changed:
            self.on_config_changed()

    def _on_device_change(self, e: ft.ControlEvent) -> None:
        self.compute_dropdown.options = _compute_options(self.device_dropdown.value or "auto")
        self.compute_dropdown.value = self.compute_dropdown.options[0].key
        self.compute_dropdown.update()
        self._notify_config_change(e)

    def _refresh_input_devices(self, _e: ft.ControlEvent) -> None:
        current = self.input_device_dropdown.value
        self.input_device_dropdown.options = self._list_input_devices()
        keys = {o.key for o in self.input_device_dropdown.options}
        self.input_device_dropdown.value = current if current in keys else "default"
        self.input_device_dropdown.update()
        if self.on_config_changed:
            self.on_config_changed()

    def _list_input_devices(self) -> list[ft.dropdown.Option]:
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

    def get_current_config(self, cache_dir: Any) -> ModelConfig:
        model_size = self.model_dropdown.value or "base"
        device = self.device_dropdown.value or "auto"
        compute_type = self.compute_dropdown.value
        batch_size = 8
        best_of = 5
        if model_size.startswith("large"):
            batch_size = 4
            best_of = 6
        return ModelConfig(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            cache_dir=cache_dir,
            batch_size=batch_size,
            best_of=best_of,
        )

    def get_audio_settings(self) -> dict[str, Any]:
        """Return dict with keys: device, capture_sample_rate, latency, blocksize"""
        device_value = self.input_device_dropdown.value
        selected_device: int | str | None
        if device_value and device_value != "default":
            try:
                selected_device = int(device_value)
            except ValueError:
                selected_device = device_value
        else:
            selected_device = None

        try:
            capture_sr = int(self.capture_sample_rate_dropdown.value or DEFAULT_CAPTURE_SAMPLE_RATE)
        except Exception:
            capture_sr = DEFAULT_CAPTURE_SAMPLE_RATE

        latency_value: str | float | None
        if (self.latency_dropdown.value or "default") == "default":
            latency_value = None
        else:
            latency_value = self.latency_dropdown.value

        blocksize_value: int | None
        if (self.blocksize_dropdown.value or "auto") == "auto":
            blocksize_value = None
        else:
            try:
                bs = self.blocksize_dropdown.value
                blocksize_value = int(bs) if bs is not None else None
            except Exception:
                blocksize_value = None
        
        return {
            "device": selected_device,
            "capture_sample_rate": capture_sr,
            "latency": latency_value,
            "blocksize": blocksize_value,
        }
