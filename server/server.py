from typing import Dict

import datetime
from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, send, emit

import numpy as np

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")


@app.route('/')
def hello_world():
    return jsonify({'Description': 'Main Page of the trainer web gui'})


@app.route('/logs/', methods=['GET'])
def get_logs():
    res = [
        {'time': datetime.datetime.now(), 'title': 'plot1', 'type': 'text', 'payload': "Hello"},
        {'time': datetime.datetime.now(), 'title': 'plot1', 'type': 'arr', 'payload': '/big_bin/id1'}
    ]
    print('answering')
    return jsonify(res)


@socketio.on('log_broadcast')
def log_message(content: str) -> None:
    print(f'Broadcasting {content}')
    emit('log', content, broadcast=True)


@socketio.on('connect')
def handle_user():
    print('A connection happened')


if __name__ == '__main__':
    # app.run()
    socketio.run(app)
