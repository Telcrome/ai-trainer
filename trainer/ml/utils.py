from enum import Enum
from typing import Generator, Tuple, Iterable, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.ndimage.morphology import distance_transform_edt as dist_trans


class ImageNormalizations(Enum):
    UnitRange = 1


def distance_transformed(mask: np.ndarray) -> np.ndarray:
    if mask.dtype != np.bool:
        mask = mask.astype(np.bool)
    return dist_trans(np.invert(mask).astype(np.float32))


def one_hot_to_cont(x: np.ndarray) -> np.ndarray:
    """
    Convert a one hot encoded image into the same image with integer representations.

    :param x: np.ndarray with (C, W, H)
    :return: np.ndarray with (W, H)
    """
    return np.argmax(x, axis=len(x.shape) - 3)


def normalize_im(im: np.ndarray, norm_type=ImageNormalizations.UnitRange) -> np.ndarray:
    """
    Currently just normalizes an image with pixel intensities in range [0, 255] to [-1, 1]
    :return: The normalized image
    """
    if norm_type == ImageNormalizations.UnitRange:
        return (im.astype(np.float32) / 127.5) - 1
    else:
        raise Exception("Unknown Normalization type")


def reduce_by_attention(arr: np.ndarray, att: np.ndarray):
    """
    Reduce an array by a field of attention, such that the result is a rectangle with the empty borders cropped.

    :param arr: Target array. The last two dimensions need to be of the same shape as the attention field
    :param att: field of attention
    :return: cropped array
    """
    assert arr.shape[-2] == att.shape[0] and arr.shape[-1] == att.shape[1]
    ones = np.argwhere(att)
    lmost, rmost = np.min(ones[:, 0]), np.max(ones[:, 0]) + 1
    bmost, tmost = np.min(ones[:, 1]), np.max(ones[:, 1]) + 1
    grid_slice = [slice(None) for _ in range(len(arr.shape) - 2)]
    grid_slice.extend([slice(lmost, rmost), slice(bmost, tmost)])
    return arr[tuple(grid_slice)], att[lmost:rmost, bmost:tmost], (lmost, rmost, bmost, tmost)


def pair_augmentation(g: Iterable[Tuple[np.ndarray, np.ndarray]], aug_ls) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    import imgaug.augmenters as iaa
    seq = iaa.Sequential(aug_ls)
    for im, gt, frame_number in g:
        im_prep = im[frame_number] if im.shape[3] > 1 else im.squeeze()
        gt_prep = np.expand_dims(gt, len(gt.shape))
        images_aug = seq(images=[im_prep], segmentation_maps=[gt_prep])
        yield images_aug[0][0].astype(np.float32), images_aug[1][0][:, :, 0].astype(np.float32), frame_number
