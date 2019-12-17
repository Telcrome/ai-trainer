import numpy as np
from PyQt5 import QtGui, QtWidgets


def pos_from_event(e):
    return int(e.scenePos().x()), int(e.scenePos().y())


def arr_to_pixmap(arr: np.ndarray) -> QtGui.QPixmap:
    image = QtGui.QImage(
        arr, arr.shape[1], arr.shape[0], arr.shape[1] * 3, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap(image)


def open_file_dialog(headline="Select a file", parent=None, dir_name=None):
    """Select a file via a dialog and return the file name."""
    if dir_name is None: dir_name = '/'
    fname = QtWidgets.QFileDialog.getOpenFileName(parent, headline,
                                                  dir_name, filter="All files (*);; SM Files (*.sm)")
    return fname[0]


def open_folder_dialog(parent=None, headline="Select directory") -> str:
    return str(QtWidgets.QFileDialog.getExistingDirectory(parent, headline))
