import datetime
from flask import Flask, jsonify
from flask_cors import CORS

import numpy as np

app = Flask(__name__)
CORS(app)

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


if __name__ == '__main__':
    app.run()
