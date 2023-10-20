from flask import Flask
from .routes.upload import upload_blueprint

def create_app():
    app = Flask(__name__)
    app.register_blueprint(upload_blueprint)
    return app
