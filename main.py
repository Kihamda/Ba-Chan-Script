from app import create_app, socketio


def main():
    """アプリケーションのエントリーポイント"""
    app = create_app()
    print("=" * 60)
    print("Faster-Whisper 文字起こしアプリを起動します")
    print("ブラウザで http://127.0.0.1:5000 を開いてください")
    print("=" * 60)
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
