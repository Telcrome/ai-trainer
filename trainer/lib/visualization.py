from __future__ import annotations

from typing import Any, List, Optional
import logging
import os
from flexx import flx

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.plotting import figure

# import trainer.lib as lib

x = np.linspace(0, 6, 50)
p1 = figure()
p1.line(x, np.sin(x))

p2 = figure()
p2.line(x, np.cos(x))


class DebuggerGui(flx.PyWidget):

    # def __init__(self):
    #     # self.bs: List[Optional[flx.Button]] = [None, None, None]
    #     super().__init__()

    def init(self):
        with flx.HBox():
            with flx.VBox():
                self.bs0 = flx.Button(text='Button1', flex=0)
                self.bs1 = flx.Button(text='Button2', flex=1)
                self.bs2 = flx.Button(text='Button3', flex=2)
                self.prog = flx.ProgressBar(flex=1, value=0.1, text='{percent} done')
                self.lbl_placeholder = flx.Label(flex=1, style='overflow-y: scroll;')
            with flx.VBox():
                self.lbl = flx.Label(flex=1, style='overflow-y: scroll;')
                # flx.BokehWidget.from_plot(p1)
                # flx.BokehWidget.from_plot(p2)


class LogWriter:

    def __init__(self, log_dir: str = './logs', id_hint='log'):
        self.prepped = False
        self.log_dir, self.log_id, self.logger = log_dir, lib.create_identifier(hint=id_hint), None

    def _prep(self):
        if not self.prepped:
            if not os.path.exists(self.log_dir):
                os.mkdir(self.log_dir)
            if not os.path.exists(self._get_run_path()):
                os.mkdir(self._get_run_path())

            # Logging to file
            self.logger = logging.getLogger('spam')
            self.logger.setLevel(logging.DEBUG)
            fh = logging.FileHandler(os.path.join(self._get_run_path(), 'logs.txt'))
            fh.setLevel(logging.DEBUG)
            self.logger.addHandler(fh)
            self.logger.propagate = False
            logging.info(self.log_id)
        self.prepped = True

    def _log_str(self, c: str):
        self._prep()
        self.logger.info(c)

    def _save_fig(self, fig: plt.Figure, title='fig', close=True, max_file_path_len=140):
        self._prep()

        f_name = f"{lib.slugify(title)}"[:max_file_path_len]
        if f_name in os.listdir(self._get_run_path()):
            f_name = f'{f_name}_{len(os.listdir(self._get_run_path()))}'
        fig.savefig(os.path.join(self._get_run_path(), f'{f_name}.png'))
        if close:
            plt.close(fig)

    def _debug_arr(self, arr: np.ndarray) -> plt.Figure:
        self._log_str(f'{arr.shape}, {arr.dtype}. Values: {np.unique(arr, return_counts=True)}')
        if len(arr.shape) == 2:
            fig, ax = plt.subplots()
            sns.heatmap(arr.astype(np.float32), ax=ax)
            return fig

    def debug_var(self, o: Any) -> None:
        """
        Allows to inspect an arbitrary type on disk.

        For saving an array with a description debug a (np.ndarray, str) tuple.

        :param o: Any variable
        """
        if isinstance(o, tuple):
            if isinstance(o[0], np.ndarray) and isinstance(o[1], str):
                fig = self._debug_arr(o[0])
                fig.suptitle(o[1])
                self._save_fig(fig, title=o[1])
            else:
                for item in o:
                    self.debug_var(item)
        elif isinstance(o, np.ndarray):
            fig = self._debug_arr(o)
            self._save_fig(fig)
        elif isinstance(o, str):
            self._log_str(f'\n{o}\n')
        else:
            self._log_str(str(o))

    def _get_run_path(self) -> str:
        return os.path.join(self.get_parent_log_folder(), self.log_id)

    def get_parent_log_folder(self) -> str:
        # self._prep()
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        return self.log_dir

    def get_absolute_run_folder(self) -> str:
        # self._prep()
        if not os.path.exists(self._get_run_path()):
            os.mkdir(self._get_run_path())
        return self._get_run_path()


logger = flx.App(DebuggerGui)
logger.export('logger.html', link=0)

if __name__ == '__main__':
    logger.launch('browser')
    flx.run()
