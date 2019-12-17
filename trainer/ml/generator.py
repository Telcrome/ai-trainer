from enum import Enum
from typing import Generator, Tuple, Iterable

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from scipy.ndimage.morphology import distance_transform_edt as dist_trans

from trainer.ml.Dataset import Dataset
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


def gray_to_rgb(im):
    im_stacked = np.zeros((im.shape[0], im.shape[1], 3))
    im_stacked[:, :, 0] = im
    im_stacked[:, :, 1] = im
    im_stacked[:, :, 2] = im
    return im_stacked


def append_dim(g: Iterable[Tuple[np.ndarray, np.ndarray]]) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    f = lambda arr: np.expand_dims(arr, axis=len(arr.shape))
    for im, gt in g:
        yield f(im), f(gt)


def distance_transformed(g: Generator) -> Generator:
    for im, gt in g:
        yield im, dist_trans(np.invert(gt.astype(np.bool)).astype(np.float32))


def resize(g: Iterable[Tuple[np.ndarray, np.ndarray]], s: Tuple[int, int]) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    for im, gt in g:
        im_resized = cv2.resize(im, s)
        gt_resized = cv2.resize(gt, s)
        yield im_resized, gt_resized


def batcherize(g: Iterable[Tuple[np.ndarray, np.ndarray]], batchsize=8) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """
    Note that all images have to be of the same size.
    :param g:
    :param batchsize:
    :return:
    """
    ims, gts = [], []
    for im, gt in g:
        if len(ims) == batchsize:
            res = np.array(ims), np.array(gts)
            yield res
            ims, gts = [], []
        ims.append(im)
        gts.append(gt)


def extract_segmentation_pair(g: Iterable[Subject], src_name: str) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    for te in g:
        res = te.get_grayimage_training_tuple_raw(src_name)
        yield res


def pair_augmentation(g: Iterable[Tuple[np.ndarray, np.ndarray]], aug_ls) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    import imgaug.augmenters as iaa
    seq = iaa.Sequential(aug_ls)
    for im, gt in g:
        images_aug = seq(images=[im], segmentation_maps=[gt.astype(np.bool)])
        res = images_aug[0][0].astype(np.float32), images_aug[1][0][:, :, 0].astype(np.float32)
        yield res


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


if __name__ == '__main__':
    d = Dataset.from_disk("C:\\Users\\rapha\\Desktop\\dataset_folder\\all_manual_train")

    g = d.random_subject_generator(split="train")
    g = extract_segmentation_pair(g)
    g = pair_augmentation(g, [
        iaa.Crop(px=(1, 16), keep_size=False),
        iaa.Fliplr(0.5),
        iaa.GammaContrast((0.5, 1.5)),
        iaa.Sometimes(0.5,
                      iaa.GaussianBlur(sigma=(0, 3.0))
                      ),
    ])
    g = resize(g, (384, 384))
    g = append_dim(batcherize(g))
    o = next(g)
