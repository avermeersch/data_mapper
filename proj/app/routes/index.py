from flask import render_template
from proj.app.app import app

@app.route('/')
def index():
    return render_template('index.html')
