# Ba-Chan Script (速攻文字起こし)

Flet + faster-whisper を使用した、ネイティブ/ウェブ両対応のハイパフォーマンス文字起こしアプリです。
リアルタイム録音と長時間ファイルの文字起こしを、GPU/CPU の力を最大限引き出して実行します。

## ✨ 主な機能

### 🎤 リアルタイム録音 & 文字起こし
- マイク入力からの音声をリアルタイムにテキスト化
- 直近 30 秒のコンテキストを表示
- 録音終了時に自動でファイル保存 & 編集モードへ移行

### 📂 ファイル文字起こし & 編集
- **無制限長対応**: 何時間の音声でもストリーミング処理で安定して文字起こし
- **編集機能**:
    - 波形を見ながら（現在は数値指定）トリミング
    - **ノイズ除去**: AI ベースのノイズ除去フィルタ搭載
    - 再エンコード（WAV / MP3 / FLAC）
    - 文字起こし結果の保存（TXT / SRT）
- **高速処理**: `faster-whisper` エンジンにより、通常の実時間の数倍〜数十倍の速度で解析

### ⚙️ 高度な設定
- **モデル選択**: `tiny` から `large-v3` まで、精度と速度のバランスを自在に選択
- **ハードウェア加速**: NVIDIA GPU (CUDA) 対応。CPU モードでも `int8` 量子化で高速化
- **音声設定**: サンプルレート、レイテンシ、ブロックサイズを詳細にチューニング可能
- **事前ロード**: 起動時または手動でモデルをウォームアップして初回推論遅延を削減

## 🚀 セットアップと実行

### 前提条件
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (推奨) または pip

### インストール & 起動

```bash
# デスクトップアプリとして起動
uv run flet run

# Web アプリとして起動
uv run flet run --web
```

初回起動時にモデルのダウンロードが行われます（`models/` ディレクトリにキャッシュ）。

## 📦 ビルド

各プラットフォーム向けのバイナリ生成には `flet build` を使用します。

```bash
flet build windows -v
flet build macos -v
flet build apk -v
```

詳細: [Flet Publishing Guide](https://docs.flet.dev/publish/)

## 🛠️ 技術スタック

- **UI**: [Flet](https://flet.dev/) (Flutter for Python)
- **AI Engine**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2)
- **Audio Processing**: `sounddevice`, `numpy`, `scipy`, `noisereduce`

---
Copyright (C) 2026 Ba-Chan Script Team
