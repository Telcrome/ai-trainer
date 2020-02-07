from typing import Callable, Tuple

import torch
from torch.utils import data
import numpy as np
from tqdm import tqdm

import trainer.ml as ml
from trainer.ml.seg_network import SegNetwork
from trainer.ml.torch_utils import device, TorchDataset


if __name__ == '__main__':
    ds = ml.Dataset.from_disk('./data/full_ultrasound_seg_0_0_9')
    seg_network = SegNetwork(
        'segi',
        in_channels=3,
        n_classes=2,
        ds=ds
    )
    train_loader = data.DataLoader(seg_network.get_torch_dataset(split='train'), batch_size=8)

    for id, (x, y) in tqdm(enumerate(train_loader)):
        # print(id)
        # print(x.shape)
        # print(y.shape)
        break

    print("finished")
