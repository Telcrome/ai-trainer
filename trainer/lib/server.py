from flask import Flask, jsonify

app = Flask(__name__)


@app.route('/')
def hello_world():
    return jsonify({'Description': 'Main Page of the trainer web gui'})


if __name__ == '__main__':
    app.run()
