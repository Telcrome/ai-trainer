from __future__ import annotations

from typing import Any, List, Optional
import logging
import os
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import trainer.lib as lib


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
        Allows to inspect an arbitrary python object on disk.

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


logger = LogWriter()


class ProgressTracker:
    """
    Allows to track solutions over time
    """

    def __init__(self, run_desc='', file_path=''):
        """
        Creates the storage file if it does not yet exist. Loads it otherwise.

        :param run_desc: Use this name to indicate what has changed
        :param file_path: Full file path to the storage json
        """
        if not file_path:
            file_path = os.path.join(logger.get_parent_log_folder(), 'progress.json')

        self.run_desc = lib.create_identifier(hint=run_desc)
        self.file_path = file_path

        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                self.json_content = json.load(f)
        else:
            self.json_content = {}
        self.json_content[self.run_desc] = {}
        self._save()

    def _save(self):
        with open(self.file_path, 'w') as f:
            json.dump(self.json_content, f)

    def add_result(self, result_name: str, flag='success', save_after=True) -> None:
        """
        Add a new result to the current run.

        :param result_name: Value of the result.
        :param flag: Flag of the result. For example 'success' or 'fail'.
        :param save_after: Decides if the progress is saved to disk immediately.
        """
        if flag not in self.json_content[self.run_desc]:
            self.json_content[self.run_desc][flag] = []

        if result_name not in self.json_content[self.run_desc][flag]:
            self.json_content[self.run_desc][flag].append(result_name)

        if save_after:
            self._save()

    def __repr__(self) -> str:
        res = f'Progress tracker with {len(self.json_content.keys())} runs\n'
        res += f'Current run: {json.dumps(self.json_content[self.run_desc])}'
        return res
