from enum import Enum
from typing import Generator, Tuple, Iterable, Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.ndimage import label, generate_binary_structure
from scipy.ndimage.morphology import distance_transform_edt as dist_trans

import trainer.lib as lib


class ImageNormalizations(Enum):
    UnitRange = 1


def duplicate_columns(data, minoccur=2):
    ind = np.lexsort(data)
    diff = np.any(data.T[ind[1:]] != data.T[ind[:-1]], axis=1)
    edges = np.where(diff)[0] + 1
    result = np.split(ind, edges)
    result = [group for group in result if len(group) >= minoccur]
    return result


def pad(small_arr: np.ndarray, size=(30, 30)) -> np.ndarray:
    # if small_arr.shape[0] < size[0] or small_arr.shape[1] < size[1]:
    size = max(small_arr.shape[0], size[0]), max(small_arr.shape[1], size[1])
    res = np.zeros(size, dtype=np.int32)
    res[:small_arr.shape[0], :small_arr.shape[1]] = small_arr
    return res
    # else:
    #     return small_arr  # There is no need for padding


def split_into_regions(arr: np.ndarray, mode=0) -> List[np.ndarray]:
    """
    Splits an array into its coherent regions.

    :param mode: 0 for orthogonal connection, 1 for full connection
    :param arr: Numpy array with shape [W, H]
    :return: A list with length #NumberOfRegions of arrays with shape [W, H]
    """
    res = []
    if mode == 0:
        rs, num_regions = label(arr)
    elif mode == 1:
        rs, num_regions = label(arr, structure=generate_binary_structure(2, 2))
    else:
        raise Exception("Please specify a valid Neighborhood mode for split_into_regions")

    for i in range(1, num_regions + 1):
        res.append(rs == i)
    return res


def normalize_im(im: np.ndarray, norm_type=ImageNormalizations.UnitRange) -> np.ndarray:
    """
    Currently just normalizes an image with pixel intensities in range [0, 255] to [-1, 1]
    :return: The normalized image
    """
    if norm_type == ImageNormalizations.UnitRange:
        return (im.astype(np.float32) / 127.5) - 1
    else:
        raise Exception("Unknown Normalization type")


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


def insert_np_at(a1: np.ndarray, a2: np.ndarray, pos: Tuple[int, int], filter_arr=None) -> np.ndarray:
    assert len(a1.shape) == 2 and len(a2.shape) == 2
    if filter_arr is None:
        filter_arr = np.ones_like(a2).astype(np.bool)
    x, y = pos
    res = np.copy(a1)
    a1_x = slice(x, min(x + a2.shape[0], a1.shape[0]))
    a1_y = slice(y, min(y + a2.shape[1], a1.shape[1]))

    if x + a2.shape[0] <= a1.shape[0]:
        a2_x = slice(0, a2.shape[0])
    else:
        a2_x = slice(0, a1.shape[0] - (x + a2.shape[0]))

    if y + a2.shape[1] <= a1.shape[1]:
        a2_y = slice(0, a2.shape[1])
    else:
        a2_y = slice(0, a1.shape[1] - (y + a2.shape[1]))
    item_filter = filter_arr[(a2_x, a2_y)]
    assert res[(a1_x, a1_y)].shape == a2[(a2_x, a2_y)].shape
    res[(a1_x, a1_y)][item_filter] = a2[(a2_x, a2_y)][item_filter]
    return res


if __name__ == '__main__':
    fit = insert_np_at(np.ones((10, 10)), np.ones((3, 3)) * 2, (2, 3))
    too_big1 = insert_np_at(np.ones((10, 10)), np.ones((3, 10)) * 2, (2, 3))
    too_big = insert_np_at(np.ones((10, 10)), np.ones((10, 10)) * 2, (2, 3))

# def put_array(big_arr: np.ndarray, small_arr: np.ndarray, offset=(0, 0)) -> np.ndarray:
#     """
#     Puts the small array into the big array. Ignores problems and does its best to fulfill the task
#     """
#     b, t =
#     big_arr[]
#     big_arr = np.putmask(big_arr, )


# if __name__ == '__main__':
# #     a = np.zeros((10, 10))
# #     b = np.random.random((4, 4))
# #     c = put_array(a, b)
# #     lib.logger.debug_var(c)
