from __future__ import annotations

from typing import Any
import logging
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

import trainer.lib as lib
from trainer.lib import get_img_from_fig, create_identifier


def debug_arr(arr: np.ndarray) -> plt.Figure:
    print(f"Debugging array with shape: {arr.shape} of type {arr.dtype}")
    unique_values = np.unique(arr, return_counts=True)
    print(unique_values)

    if len(arr.shape) == 2:
        fig, ax = plt.subplots()
        sns.heatmap(arr.astype(np.float32), ax=ax)
        return fig


class LogWriter:

    def __init__(self, log_dir: str = './logs', id_hint='log_str'):
        self.prepped = False
        self.log_dir, self.log_id, self.logger, self.visboard = log_dir, lib.create_identifier(hint=id_hint), None, None

    def prep(self):
        if not self.prepped:
            if not os.path.exists(self.log_dir):
                os.mkdir(self.log_dir)
            os.mkdir(self.get_run_path())

            # Logging to file
            self.logger = logging.getLogger('spam')
            self.logger.setLevel(logging.DEBUG)
            fh = logging.FileHandler(os.path.join(self.get_run_path(), 'logs.txt'))
            fh.setLevel(logging.DEBUG)
            self.logger.addHandler(fh)
            self.logger.propagate = False
            logging.info(self.log_id)
        self.prepped = True

    def log_str(self, c: str):
        self.prep()
        self.logger.info(c)

    def debug_var(self, o: Any):
        if isinstance(o, np.ndarray):
            fig = debug_arr(o)
            self.save_fig(fig)
            self.log_str(f'{o.shape}, {o.dtype}. Values: {np.unique(o, return_counts=True)}')
        elif isinstance(o, str):
            self.log_str(o)
        else:
            self.log_str(str(o))

    def get_run_path(self) -> str:
        return os.path.join(self.log_dir, self.log_id)

    def add_scalar(self, tag: str, val: float, step: int):
        self.prep()
        self.visboard.writer.add_scalar(tag, val, step)

    def save_fig(self, fig: plt.Figure):
        self.prep()
        fig.savefig(os.path.join(self.get_run_path(),
                                 f"{lib.create_identifier('fig')}_{len(os.listdir(self.get_run_path()))}.png"))

    def save_tensor(self, arr: torch.Tensor, name="tensor"):
        self.prep()
        tensor_dir = os.path.join(self.get_run_path(), name)
        if not os.path.exists(tensor_dir):
            os.mkdir(tensor_dir)
        torch.save(arr, os.path.join(tensor_dir, f'{len(os.listdir(tensor_dir))}.pt'))

    def add_model(self, model: nn.Module, input_batch: torch.Tensor):
        self.prep()
        self.visboard.writer.add_graph(model, input_batch)
        # TODO: Write down some textual representation


logger = LogWriter()
