"""
Specifies utility functions, aiming to allow rapid developing of domain specific computational graphs.
"""
from typing import Callable, List, Tuple

import cv2
import numpy as np
import scipy.ndimage as ndimage

import trainer.ml as ml
from trainer.demo_data.arc import Value

ortho_filter = ndimage.generate_binary_structure(2, 1)
full_filter = ndimage.generate_binary_structure(2, 2)
diag_filter = full_filter != ortho_filter


def general_transform(grid: np.ndarray, labels: np.ndarray, f: Callable[[np.ndarray], np.ndarray]):
    """
    Applies a transformation f on every labelled region in grid independently
    """
    unique = np.unique(labels)[np.unique(labels) != 0]
    res = np.zeros_like(grid)
    for l in unique:
        reduced, reducedatt, (l, r, b, t) = ml.reduce_by_attention(grid, labels == l)
        transformed = f(reduced)
        transformed_att = f(reducedatt)
        # t_obj = transformed[transformed_att]
        # if len(t_obj.shape) < 2:
        #     print("WTF")
        assert transformed_att.shape == transformed.shape
        res = ml.insert_np_at(res, transformed, (l, b), filter_arr=transformed_att)
        # try:
        #     # res[l:transformed_att.shape[0] + l, b:transformed_att.shape[1] + b][transformed_att] = transformed[
        #     #     transformed_att]
        # except Exception as e:
        #     print(e)
        # res[l:r, b:t][transformed_att] = transformed[transformed_att]
    return res


def zoom_in(arr: np.ndarray, factor: int):
    if len(np.unique(arr)) == 1 or factor <= 0:
        return arr
    assert int(factor) == factor
    grid, reducedatt, (l, r, b, t) = ml.reduce_by_attention(arr.astype(np.uint8), arr != 0)
    resize = tuple(int(d * factor) for d in grid.shape)
    res = np.zeros_like(arr)
    out = cv2.resize(grid, resize, interpolation=cv2.INTER_NEAREST)
    res = ml.insert_np_at(res, out, (l, b))
    return res


def objects_from_labels(labels: np.ndarray) -> List[np.ndarray]:
    res = []
    for u in np.unique(labels):
        if u != 0:
            res.append((labels == u))
    return res


def select_objects(objts: Tuple[np.ndarray, np.ndarray], lbl_to_float: Callable, count_index: int, reverted: bool):
    arrs = objects_from_labels(objts[1])

    lbls = [arr.astype(np.uint8) for arr in arrs]
    if not arrs:
        # print("There were no objects")
        return objts[0], np.ones_like(objts[1]).astype(np.bool)
        # return np.random.randint(0, 2, size=(30, 30))

    i = count_index % len(arrs)
    # assert 0 < min([np.max(label) for label in labels])
    ps = [lbl_to_float(lbl) for lbl in lbls]
    sorting = np.flip(np.argsort(ps))
    res = np.zeros_like(objts[0])

    if reverted:
        mask_i = arrs[sorting[i]]
    else:
        mask_i = arrs[sorting[len(sorting) - 1 - i]]

    res[mask_i] = objts[0][mask_i]

    res[mask_i] = objts[0][mask_i]
    m = mask_i
    assert res.shape == (30, 30) and m.shape == (30, 30)
    return res, m


def get_lbl_lu(obj: np.ndarray) -> Tuple[int, int]:
    if np.max(obj) == 0:
        return 0, 0
    obj_indices = np.argwhere(obj)
    return tuple(np.min(obj_indices, axis=0))


colour_converter = {v.value: v for v in Value}
