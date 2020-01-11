import torch
import matplotlib.pyplot as plt

from trainer.bib import get_img_from_fig


class VisBoard:
    """
    Allows visualizing various information during the training process.

    Builds on Tensorboard.

    Can be invoked by the following command
    ```bash
    tensorboard --logdir=tb --samples_per_plugin=images=100
    ```
    """

    def __init__(self, run_name='run', dir_name='tb'):
        self.dir_name = dir_name
        self.run_name = run_name
        self.writer = torch.utils.tensorboard.SummaryWriter(f'{self.dir_name}/{self.run_name}')

    def add_figure(self, fig: plt.figure, group_name: str):
        img = torch.from_numpy(get_img_from_fig(fig)).permute((2, 0, 1))
        self.writer.add_image(group_name, img)
        plt.close(fig)
