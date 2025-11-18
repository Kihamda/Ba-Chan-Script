# Faster-Whisper 文字起こし Web アプリ

faster-whisper を使った高速かつ高精度な文字起こし Web アプリケーション

## 機能

### 1. リアルタイム文字起こしモード

- マイクから直接音声を録音して高速に文字起こし
- WebSocket を使ったリアルタイム処理
- 軽量モデル（base）で高速処理

### 2. ファイルアップロードモード

- 音声/動画ファイルをアップロードして高精度に文字起こし
- 大型モデル（large-v3）で最高精度
- タイムスタンプ付き文字起こし結果
- JSON 形式でのエクスポート機能

## 対応フォーマット

- MP3, WAV, MP4, M4A, WEBM, FLAC, OGG
- 最大ファイルサイズ: 500MB

## セットアップ

### 前提条件

- Python 3.11 以上
- uv（Python パッケージマネージャー）

### インストール

```powershell
# 依存パッケージのインストール（既に完了済み）
uv sync

# アプリケーション起動
uv run python main.py
```

ブラウザで `http://127.0.0.1:5000` を開く

## CUDA 対応（オプション）

GPU で高速化したい場合は CUDA 対応の PyTorch をインストール

```powershell
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 使い方

### リアルタイムモード

1. 「リアルタイム文字起こし」タブを選択
2. 言語を選択
3. 「録音開始」ボタンをクリック
4. 話す
5. 「録音停止」ボタンをクリック
6. 自動的に文字起こしが開始される

### ファイルアップロードモード

1. 「ファイル文字起こし」タブを選択
2. 言語と精度（ビームサイズ）を選択
3. 音声ファイルをドラッグ&ドロップまたは選択
4. 「文字起こし開始」ボタンをクリック
5. 処理完了後、結果が表示される
6. 必要に応じてコピーまたは JSON 形式でダウンロード

## モデルについて

### リアルタイムモード: base

- サイズ: 約 140MB
- 速度: 非常に高速
- 精度: 標準

### ファイルモード: large-v3

- サイズ: 約 3GB
- 速度: やや遅い
- 精度: 最高

初回起動時にモデルが自動ダウンロードされます（./models ディレクトリ）

## 技術スタック

- **バックエンド**: Flask, Flask-SocketIO
- **文字起こしエンジン**: faster-whisper (CTranslate2)
- **フロントエンド**: Tailwind CSS, Socket.IO
- **パッケージ管理**: uv

## パフォーマンス

### CPU（Intel Core i7-12700）

- リアルタイムモード: RTF 0.3x 程度（3 秒の音声を 1 秒で処理）
- 高精度モード: RTF 1.5x 程度（3 秒の音声を 4.5 秒で処理）

### GPU（NVIDIA RTX 3060）

- リアルタイムモード: RTF 0.1x 程度
- 高精度モード: RTF 0.4x 程度

RTF (Real-Time Factor): 音声時間に対する処理時間の比率。小さいほど高速

## ライセンス

MIT

## 参考

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [OpenAI Whisper](https://github.com/openai/whisper)
