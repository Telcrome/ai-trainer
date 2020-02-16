import logging
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

import trainer.lib as lib
from trainer.lib import get_img_from_fig, create_identifier


class LogWriter:

    def __init__(self, log_dir: str = './logs', id_hint='log'):
        self.log_dir = log_dir
        self.log_id = lib.create_identifier(hint=id_hint)
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        os.mkdir(self.get_run_path())

        # Logging to file
        self.logger = logging.getLogger('spam')
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(os.path.join(self.log_dir, 'logs.log'))
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)
        logging.info(self.log_id)

        # Logging to Tensorboard
        self.visboard = VisBoard(run_name=self.log_id, dir_name=os.path.join(self.log_dir, 'tb'))

    def log(self, c: str):
        self.logger.info(c)

    def get_run_path(self) -> str:
        return os.path.join(self.log_dir, self.log_id)

    def add_scalar(self, tag: str, val: float, step: int):
        self.visboard.writer.add_scalar(tag, val, step)

    def save_tensor(self, arr: torch.Tensor, name="tensor"):
        tensor_dir = os.path.join(self.get_run_path(), name)
        if not os.path.exists(tensor_dir):
            os.mkdir(tensor_dir)
        torch.save(arr, os.path.join(tensor_dir, f'{len(os.listdir(tensor_dir))}.pt'))


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

    def visualize_subject(self, s: lib.Subject):
        fig, ax = plt.subplots()
        raise NotImplementedError()
