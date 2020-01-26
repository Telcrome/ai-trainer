# %%

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf

import trainer.ml as ml
from trainer.ml.data_loading import get_subject_gen, random_struct_generator, get_img_mask_pair

from trainer.ml import batcherize, resize, normalize_im


def g_convert(g):
    for vid, gt, f in g:
        if f >= 2:
            res = np.zeros((vid.shape[1], vid.shape[2], 3))
            # Discard most of the video, deeplab can handle only 3 frames
            res[:, :, 0] = vid[f - 2, :, :, 0]
            res[:, :, 1] = vid[f - 1, :, :, 0]
            res[:, :, 2] = vid[f, :, :, 0]
            gt_stacked = np.zeros((gt.shape[0], gt.shape[1], 2), dtype=np.float32)
            gt_stacked[:, :, 0] = gt.astype(np.float32)
            gt_stacked[:, :, 1] = np.invert(gt).astype(gt_stacked.dtype)
            yield normalize_im(res).astype(np.float32), gt_stacked


if __name__ == '__main__':
    ds = ml.Dataset.download(url='https://rwth-aachen.sciebo.de/s/1qO95mdEjhoUBMf/download',
                             local_path='./data',  # Your local data folder
                             dataset_name='crucial_ligament_diagnosis'  # Name of the dataset
                             )

    structure_name = 'femur'  # The structure that we now train for

    # Simple generator preprocessing chain
    g_raw = random_struct_generator(ds, structure_name)
    g_extracted = g_convert(g_raw)
    g_resized = resize(g_extracted, (384, 384))
    g_train = batcherize(g_resized, batchsize=1)

    # training loop
    for x, y in g_train:
        print(x.shape)
        print(y.shape)
        break
