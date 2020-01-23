import os
import sys
from typing import Dict, Callable

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

from trainer.lib.tgui.TConsole import TConsole


class TContentGrid(QtWidgets.QWidget):
    def __init__(self, main_widget: QtWidgets.QWidget, console: TConsole):
        super(TContentGrid, self).__init__()
        layout = QtWidgets.QHBoxLayout()

        # Toolbar
        toolbar_scroller = QtWidgets.QScrollArea()
        toolbar_scroller.setFixedWidth(250)
        toolbar_scroller.setWidgetResizable(True)
        toolbar_widget = QtWidgets.QWidget()
        self.toolbar = QtWidgets.QVBoxLayout()
        self.toolbar.setAlignment(Qt.AlignTop)
        toolbar_widget.setLayout(self.toolbar)
        toolbar_scroller.setWidget(toolbar_widget)
        layout.addWidget(toolbar_scroller)

        # Main Content
        right_side = QtWidgets.QWidget()
        right_side_layout = QtWidgets.QVBoxLayout()
        # self.main_content = QtWidgets.QWidget()
        self.main_content = main_widget

        # Bottom
        self.logging_widget = QtWidgets.QPlainTextEdit("Here we will see a process log in the future")
        self.logging_widget.setReadOnly(True)
        self.logging_widget.setFrameStyle(QtWidgets.QFrame.StyledPanel)

        logging_splitter = QtWidgets.QSplitter(Qt.Vertical)
        right_side_layout.addWidget(logging_splitter)
        logging_splitter.addWidget(self.main_content)

        logging_splitter.addWidget(console)
        logging_splitter.setSizes([600, 200])

        right_side.setLayout(right_side_layout)
        layout.addWidget(right_side)

        self.setLayout(layout)

    def add_tool(self, w):
        self.toolbar.addWidget(w)


class TWindow(QtWidgets.QMainWindow):

    def __init__(self, main_widget: QtWidgets.QWidget = None, title="Window title", maximized=False):
        super(TWindow, self).__init__()

        # Allow a debug console to be used
        self.console = TConsole()

        if main_widget is not None:
            self.main_widget = main_widget
        else:
            self.main_widget = QtWidgets.QLabel("No main content is set")
        self.actions = {"Window": [
            self.create_action_from_tuple("Close Window", "Ctrl+Q", "Closes the current window", self.exit_window)
        ]}
        self.sub_windows: Dict[str, TWindow] = {}

        self.setWindowTitle(title)
        self.content_grid = None
        self.gui_initialized = False
        if maximized:
            self.showMaximized()
        print('Process ID is:', os.getpid())

    def init_ui(self, actions: Dict = None):
        if self.gui_initialized:
            return
        if actions is not None:
            for action_key in actions:
                self.actions[action_key] = actions[action_key]

        self.statusBar()
        self.create_menu_bar()
        self.content_grid = TContentGrid(self.main_widget, self.console)
        self.setCentralWidget(self.content_grid)
        self.gui_initialized = True

    def show_subwindow(self, name: str, w):
        self.sub_windows[name] = w
        w.show()

    def create_action_from_tuple(self, action_name: str, shortcut: str, status_tip: str, handler: Callable):
        action = QtWidgets.QAction(f"&{action_name}", self)
        action.setShortcut(shortcut)
        action.setStatusTip(status_tip)
        action.triggered.connect(handler)
        return action

    def exit_window(self):
        self.close()

    def create_menu_bar(self):
        main_menu = self.menuBar()
        for parent_key in self.actions:
            menu = main_menu.addMenu(f"&{parent_key}")
            for a in self.actions[parent_key]:
                menu.addAction(a)

    def get_user_input(self, headline="Headline", dialog_text="Input"):
        return QtWidgets.QInputDialog.getText(self, headline, dialog_text)


def run_window(cls, *args):
    def exception_hook(exctype, value, traceback):
        print(exctype, value, traceback)
        sys._excepthook(exctype, value, traceback)
        sys.exit(1)

    sys._excepthook = sys.excepthook
    sys.excepthook = exception_hook

    app = QtWidgets.QApplication(sys.argv)
    gui = cls(*args)
    gui.init_ui()
    gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    run_window(TWindow)
