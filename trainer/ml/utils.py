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


def append_dicom_to_subject(te_path: str,
                            dicom_path: str,
                            binary_name: str = '',
                            seg_structs: Dict[str, str] = None,
                            auto_save=True) -> Subject:
    """

    :param te_path: directory path to the subject
    :param dicom_path: filepath to the dicom containing the image data
    :param binary_name: Name of the binary, if not provided a name is chosen.
    :param seg_structs: Structures that can be segmented in the image data
    :param auto_save: The new state of the subject is automatically saved to disk
    :return: The subject containing the new data
    """
    s = Subject.from_disk(te_path)

    if not binary_name:
        binary_name = create_identifier(hint='DICOM')

    from trainer.lib.dicom_utils import import_dicom

    img_data, meta = import_dicom(dicom_path)
    s.add_source_image_by_arr(img_data, binary_name, structures=seg_structs, extra_info=meta)

    if auto_save:
        s.to_disk(s.get_parent_directory())

    return s


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
