import typing as typ
from websocket import create_connection
import socketio

import PySimpleGUI as sg
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# ws = create_connection("ws://localhost:5000/")
sio = socketio.Client()
sio.connect('http://localhost:5000')


def debug_arr(arr: np.ndarray):
    print(f"Debugging array with shape: {arr.shape} of type {arr.dtype}")
    unique_values = np.unique(arr, return_counts=True)
    print(unique_values)

    if len(arr.shape) == 2:
        sns.heatmap(arr.astype(np.float32))
        plt.show()


def debug_var(o: typ.Any):
    if isinstance(o, np.ndarray):
        debug_arr(o)
    elif isinstance(o, str):
        sio.emit('log_broadcast',
                 {
                     'string': o
                 })


if __name__ == '__main__':
    test_arr = np.random.random((5, 5))

    debug_var("Starting debugging application")


    def draw_plot():
        plt.plot([0.1, 0.2, 0.5, 0.7])
        plt.show(block=False)


    layout = [
        [sg.Canvas(size=(640, 480), key='canvas')],
        [sg.Button('Plot'), sg.Cancel(), sg.Button('Popup')]
    ]

    window = sg.Window('Have some Matplotlib....', layout)

    while True:
        event, values = window.read()
        if event in (None, 'Cancel'):
            break
        elif event == 'Plot':
            draw_plot()
        elif event == 'Popup':
            # sg.popup('Yes, your application is still running')
            debug_var('pressed a button')
    window.close()
