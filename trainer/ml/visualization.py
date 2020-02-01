import os

import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

from trainer.lib import get_img_from_fig, create_identifier
import trainer.ml as ml
import trainer.lib as lib


class LogWriter:

    def __init__(self, log_dir: str = './logs', id_hint='log'):
        self.log_dir = log_dir
        self.log_id = lib.create_identifier(hint=id_hint)

    def get_run_path(self) -> str:
        return os.path.join(self.log_dir, self.log_id)

    def save_tensor(self, arr: torch.Tensor, name="tensor"):
        torch.save(arr, os.path.join(self.get_run_path(), f'{name}.pt'))


class VisBoard:
    """
    Allows visualizing various information during the training process.

    Builds on Tensorboard.

    Can be invoked by the following command
    ```bash
    tensorboard --logdir=tb --samples_per_plugin=images=100
    ```
    """

    def __init__(self, run_name='', dir_name='tb'):
        self.dir_name = dir_name
        self.run_name = run_name
        if not self.run_name:
            self.run_name = create_identifier()
        self.writer = SummaryWriter(f'{self.dir_name}/{self.run_name}')

    def add_scalar(self, tag: str, val: float, step: int):
        self.writer.add_scalar(tag, val, step)

    def add_figure(self, fig: plt.Figure, group_name: str = "Default", close_figure: bool = True) -> None:
        """
        Logs a matplotlib.pyplot Figure.
        @param fig: The figure to be logged
        @param group_name: Multiple figures can be grouped by using an identical group name
        @param close_figure: If the figure should not be closed specify False
        """
        img = torch.from_numpy(get_img_from_fig(fig)).permute((2, 0, 1))
        self.writer.add_image(group_name, img)
        if close_figure:
            plt.close(fig)

    def visualize_subject(self, s: ml.Subject):
        fig, ax = plt.subplots()
