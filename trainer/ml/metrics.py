from typing import Union

import numpy as np
import torch


class SegmentationMetric:
    """
    The class takes two numpy arrays or torch tensors as input and computes various segmentation metrics.


    """

    def __init__(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]):
        assert (x.dtype == np.bool and y.dtype == np.bool and len(x.shape) == 2 and len(y.shape) == 2)
        self.x, self.y = x, y

    def compute_iou(self):
        pass

    def compute_dice(self):
        pass

    def compute_hausdorff(self):
        pass


if __name__ == '__main__':
    print('.')
