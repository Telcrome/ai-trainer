"""
A server that is concerned with the manipulation of one dataset at a time.
"""

from flask import Flask, render_template

from trainer import VERSION

app = Flask(__name__)
print(__name__)


@app.route('/')
def angular_frontend():
    return render_template(template_name_or_list='main.html', version=VERSION)


@app.route('/version')
def get_version():
    return {
        "version": VERSION
    }
