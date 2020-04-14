import datetime
from flask import Flask, jsonify

import numpy as np

app = Flask(__name__)


@app.route('/')
def hello_world():
    return jsonify({'Description': 'Main Page of the trainer web gui'})


@app.route('/logs/', methods=['GET'])
def get_logs():
    res = [
        {'time': datetime.datetime.now(), 'title': 'plot1', 'type': 'text', 'payload': "Hello"},
        {'time': datetime.datetime.now(), 'title': 'plot1', 'type': 'arr', 'payload': '/big_bin/id1'}
    ]
    return jsonify(res)


if __name__ == '__main__':
    app.run()
