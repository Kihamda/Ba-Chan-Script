from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
import os

socketio = SocketIO()


def create_app():
    """Flaskアプリケーションファクトリ"""
    app = Flask(__name__)
    
    # 設定
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
    
    # アップロードフォルダ作成
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # CORS有効化
    CORS(app)
    
    # SocketIO初期化
    socketio.init_app(app, cors_allowed_origins="*", async_mode='threading')
    
    # ルート登録
    from app import routes
    app.register_blueprint(routes.bp)
    
    return app
