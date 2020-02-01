from enum import Enum
from typing import Generator, Tuple, Iterable, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.ndimage.morphology import distance_transform_edt as dist_trans

from trainer.lib import create_identifier
from trainer.ml import Subject


class ImageNormalizations(Enum):
    UnitRange = 1


def normalize_im(im: np.ndarray, norm_type=ImageNormalizations.UnitRange):
    """
    Currently just normalizes an image with pixel intensities in range [0, 255] to [-1, 1]
    :return: The normalized image
    """
    if norm_type == ImageNormalizations.UnitRange:
        return (im.astype(np.float32) / 127.5) - 1
    else:
        raise Exception("Unknown Normalization type")


def append_dim(g: Iterable[Tuple[np.ndarray, np.ndarray]]) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    f = lambda arr: np.expand_dims(arr, axis=len(arr.shape))
    for im, gt in g:
        yield f(im), f(gt)


def distance_transformed(g: Generator) -> Generator:
    for im, gt in g:
        yield im, dist_trans(np.invert(gt.astype(np.bool)).astype(np.float32))


def pair_augmentation(g: Iterable[Tuple[np.ndarray, np.ndarray]], aug_ls) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    import imgaug.augmenters as iaa
    seq = iaa.Sequential(aug_ls)
    for im, gt, frame_number in g:
        im_prep = im[frame_number] if im.shape[3] > 1 else im.squeeze()
        gt_prep = np.expand_dims(gt, len(gt.shape))
        images_aug = seq(images=[im_prep], segmentation_maps=[gt_prep])
        yield images_aug[0][0].astype(np.float32), images_aug[1][0][:, :, 0].astype(np.float32), frame_number


def visualize_batch(t: Tuple[np.ndarray, np.ndarray]) -> None:
    im_arr, gt_arr = t
    ims = [im_arr[i, :, :, 0] for i in range(im_arr.shape[0])]
    gts = [gt_arr[i, :, :, 0] for i in range(gt_arr.shape[0])]
    for i, _ in enumerate(ims):
        plt.subplot(1, 2, 1)
        sns.heatmap(ims[i])
        plt.subplot(1, 2, 2)
        sns.heatmap(gts[i])
        plt.show()
