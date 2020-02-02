from typing import Callable, Tuple

from torch.utils import data
import numpy as np
from tqdm import tqdm

import trainer.ml as ml
from trainer.ml.seg_network import SegNetwork


class TorchDataset(data.Dataset):

    def __init__(self,
                 ds_path: str,
                 f: Callable[[ml.Subject], Tuple[np.ndarray, np.ndarray]],
                 split=''):
        super().__init__()
        self.ds = ml.Dataset.from_disk(ds_path)
        self.preprocessor = f
        self.split = split
        self.ss = self.ds.get_subject_name_list(split=self.split)

    def __getitem__(self, item):
        # print(f'item: {item}')
        s = self.ds.get_subject_by_name(self.ss[item])
        return self.preprocessor(s)

    def __len__(self):
        return self.ds.get_subject_count(split=self.split)


if __name__ == '__main__':
    td = TorchDataset('./data/full_ultrasound_seg_0_0_9', SegNetwork.preprocess)
    train_loader = data.DataLoader(td, batch_size=8, num_workers=4)

    for id, (x, y) in tqdm(enumerate(train_loader)):
        # print(id)
        # print(x.shape)
        # print(y.shape)
        pass

    print("finished")
