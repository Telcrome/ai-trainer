from __future__ import annotations
import itertools
import random
from enum import Enum
from abc import ABC
from typing import TypeVar, NewType, Union, Callable, Tuple, Any, Dict, get_type_hints, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import pandas as pd
from scipy.stats import mode
import scipy.ndimage as ndimage
from scipy.ndimage.interpolation import shift
from scipy.signal import convolve2d
from skimage.filters import rank_order
import skimage.measure as sk_measure
from skimage.metrics import structural_similarity as ssim
from skimage.future import manual_lasso_segmentation
from scipy.ndimage import label
from skimage.measure import block_reduce
import skimage.filters.rank as ranks
from skimage.transform import downscale_local_mean

import trainer.lib as lib
import trainer.ml as ml
from trainer.demo_data.arc import Value, plot_as_heatmap, game_from_subject
from trainer.cg.dsl_utils import general_transform, ortho_filter, full_filter, diag_filter, colour_converter, zoom_in, \
    objects_from_labels, select_objects, get_lbl_lu
from trainer.cg.samplers import RandomInteger

BoolFilter = NewType('BoolFilter', np.ndarray)

ValueGrid = NewType('ValueGrid', np.ndarray)
BoolGrid = NewType('BoolGrid', np.ndarray)
IntGrid = NewType('IntGrid', np.ndarray)
RealGrid = NewType('RealGrid', np.ndarray)
ObjectLabels = NewType('ObjectLabels', Tuple[np.ndarray, np.ndarray])
SingleObject = NewType('SingleObject', Tuple[np.ndarray, np.ndarray])

IntLine = NewType('IntLine', np.ndarray)

Position = NewType('Position', Tuple[int, int])
Offset = NewType('Offset', Tuple[int, int])
NonZeroNumber = NewType('NonZeroNumber', int)


class Orientation(Enum):
    horizontal, vertical, diagonal_dota, diagonal_other = range(4)


class B(Enum):
    T, F = range(2)


def is_value(arr: ValueGrid, v: Value) -> BoolGrid:
    return arr == v.value


def value_to_arr(v: Value) -> ValueGrid:
    arr = np.zeros((30, 30))
    arr[:, :] = v.value
    return arr


def negated_arr(arr: BoolGrid, negated: B) -> BoolGrid:
    if negated == B.T:
        return ~arr
    else:
        return arr


def sorted_values(grid: ValueGrid) -> List[Value]:
    numbers, counts = np.unique(grid, return_counts=True)
    c_sort = np.flip(np.argsort(counts))
    res = list(numbers[c_sort])
    v_res = [colour_converter[v] for v in res]
    return v_res


def pick_from_values(values: List[Value], count_index: RandomInteger, reverted: B) -> Value:
    i = count_index % len(values)
    if reverted == B.F:
        return values[i]
    else:
        return values[len(values) - 1 - i]


def coord(grid: ValueGrid, modulo: RandomInteger) -> IntLine:
    output = np.arange(grid.shape[0])
    if modulo != 0:
        return np.mod(output, modulo)
    else:
        return output


def different_cells_in_line(grid: ValueGrid, orientation: Orientation) -> IntLine:
    output = np.zeros(grid.shape[0])

    # Populate output with the number of different colors in that particular line
    if orientation == Orientation.vertical:
        for i in range(grid.shape[1]):
            output[i] = len(np.unique(grid[:, i]))
    elif orientation == Orientation.horizontal:
        for i in range(grid.shape[0]):
            output[i] = len(np.unique(grid[i, :]))
    return output


def arr_from_line(line: IntLine, orientation: Orientation) -> IntGrid:
    output = np.zeros((line.shape[0], line.shape[0]))

    if orientation == Orientation.vertical or orientation == Orientation.diagonal_dota:
        for i in range(line.shape[0]):
            output[:, i] = line[i]
    else:
        for i in range(line.shape[0]):
            output[i, :] = line[i]

    return output


###############################################
# Objectness
###############################################

class RegionProp(Enum):
    area, bbox_area, convex_area, eccentricity, equivalent_diameter, euler_number, extent, filled_area = range(8)
    centroid_row, centroid_column, major_axis_length, minor_axis_length, orientation, perimeter, solidity = range(9, 16)


def compute_region_prop(prop, rp: RegionProp) -> float:
    if rp == RegionProp.area:
        p = prop.area
    elif rp == RegionProp.bbox_area:
        p = prop.bbox_area
    elif rp == RegionProp.convex_area:
        p = prop.convex_area
    elif rp == RegionProp.eccentricity:
        p = prop.eccentricity
    elif rp == RegionProp.equivalent_diameter:
        p = prop.equivalent_diameter
    elif rp == RegionProp.euler_number:
        p = prop.euler_number
    elif rp == RegionProp.extent:
        p = prop.extent
    elif rp == RegionProp.filled_area:
        p = prop.filled_area
    elif rp == RegionProp.centroid_row:
        p = prop.centroid[0]
    elif rp == RegionProp.centroid_column:
        p = prop.centroid[1]
    elif rp == RegionProp.major_axis_length:
        p = prop.major_axis_length
    elif rp == RegionProp.minor_axis_length:
        p = prop.major_axis_length
    elif rp == RegionProp.orientation:
        p = prop.orientation
    elif rp == RegionProp.perimeter:
        p = prop.perimeter
    elif rp == RegionProp.solidity:
        p = prop.solidity
    else:
        raise Exception(f"rp has an invalid value {rp}")
    return p


def reg_quantity(labels: ObjectLabels, rp: RegionProp) -> RealGrid:
    # First label connected regions
    # Then output properties of each region
    props = sk_measure.regionprops(labels[1])
    res = np.zeros_like(labels[0])
    for i, prop in enumerate(props):
        res[labels[1] == i + 1] = compute_region_prop(prop, rp)
    return res


class Structs(Enum):
    Orthogonal, Full = 1, 2


def lbl_connected(arr: ValueGrid, structure: Structs, background: Value) -> ObjectLabels:
    bg = background.value
    # labels, num = label(arr, structure=structure)
    labels = sk_measure.label(arr, background=bg, connectivity=structure.value)
    return arr, labels


def lbl_by_bg(arr: ValueGrid, structure: Structs, background: Value) -> ObjectLabels:
    bg_arr = ((arr != background.value) & (arr != 0)).astype(np.uint8)
    labels = sk_measure.label(bg_arr, background=0, connectivity=structure.value)
    return arr, labels


def lbl_by_bool(grid: ValueGrid, arr: BoolGrid) -> ObjectLabels:
    return grid, arr.astype(np.uint8)


def object_by_ordering(objts: ObjectLabels, prop: RegionProp, count_index: RandomInteger, reverted: B) -> SingleObject:
    res = select_objects(objts, lambda lbl: compute_region_prop(sk_measure.regionprops(lbl)[0], prop), count_index,
                         reverted == B.F)
    return res


def origin() -> Position:
    return 0, 0


def get_obj_lu(obj: SingleObject) -> Position:
    """
    :param grid: Zero-padded object
    :return: (x, y) coordinate of the left upper corner of a zero-padded object
    """
    if np.max(obj[1]) == 0:
        return 0, 0
    obj_indices = np.argwhere(obj[1])
    return tuple(np.min(obj_indices, axis=0))


def object_by_spatial(objts: ObjectLabels, count_index: RandomInteger, reverted: B) -> SingleObject:
    def lbl_to_priority(lbl: np.ndarray):
        x, y = get_lbl_lu(lbl)
        return x + y * 30

    # pos_to_priority = lambda x: x[0] + x[1] * 30
    res = select_objects(objts, lbl_to_priority, count_index, reverted == B.F)
    return res


def move_to(obj: SingleObject, location: Position) -> SingleObject:
    # If there is no Empty space, moving makes no sense even though its a valid operation
    assert obj[0].shape == (30, 30) and obj[1].shape == (30, 30)
    if np.max(np.unique(obj[1])) == 0:
        return obj
    reduced, reducedatt, (l, r, b, t) = ml.reduce_by_attention(obj[0], obj[1])
    res, res_mask = np.zeros_like(obj[0]), np.zeros_like(obj[1])
    res = ml.insert_np_at(res, reduced, location)
    res_mask = ml.insert_np_at(res_mask, reducedatt, location)
    return res, res_mask


###############################################
# Numbers
###############################################

def non_zero_num(x: RandomInteger, b: B) -> NonZeroNumber:
    if b == B.T:
        return x
    else:
        return -1 * x


def measure_grid(grid: ValueGrid, orientation: Orientation) -> RandomInteger:
    if len(np.unique(grid)) == 1:
        # return np.random.random(grid.shape)
        return random.randint(0, 30)
    reduced, reducedatt, (l, r, b, t) = ml.reduce_by_attention(grid, grid != 0)
    if orientation == Orientation.horizontal or orientation == Orientation.diagonal_dota:
        return reduced.shape[1]
    else:
        return reduced.shape[0]


def measure_grid_b(grid: BoolGrid, orientation: Orientation) -> RandomInteger:
    return measure_grid(grid, orientation)


def hist(grid: ValueGrid) -> IntGrid:
    vals, nums = np.unique(grid, return_counts=True)
    res = np.zeros_like(grid, dtype=np.int)
    for i, v in enumerate(vals):
        res[grid == v] = nums[i]
    return res


###############################################
# Everything concerned with transformations
###############################################

class OneShotTransform(Enum):
    Identity, Rot90, Rot180, Rot270 = range(4)
    FlipLR, FlipUD = range(10, 12)


def transform(labelling: ObjectLabels, t_type: OneShotTransform) -> ValueGrid:
    """
    Every coherent region is transformed independently.

    If objects are classified by their color it works fine to transform the input directly.
    If not first foreground needs to be separated from background by a labelling strategy.
    """

    if t_type == OneShotTransform.Rot90:
        f = np.rot90
    elif t_type == OneShotTransform.Rot180:
        f = lambda x: np.rot90(x, k=2)
    elif t_type == OneShotTransform.Rot270:
        f = lambda x: np.rot90(x, k=3)
    elif t_type == OneShotTransform.FlipLR:
        f = np.fliplr
    elif t_type == OneShotTransform.FlipUD:
        f = np.flipud
    else:
        assert t_type == OneShotTransform.Identity
        # f = lambda x: x
        return labelling[0]
    return general_transform(labelling[0], labels=labelling[1], f=f)


def direction_step(step_size: NonZeroNumber, direction: Orientation) -> Offset:
    if direction == Orientation.horizontal:
        return 0, step_size
    elif direction == Orientation.vertical:
        return step_size, 0
    elif direction == Orientation.diagonal_dota:
        return -1 * step_size, step_size
    else:
        return step_size, step_size


def make_offset(x: NonZeroNumber, y: NonZeroNumber) -> Offset:
    return x, y


def shift_val_arr(grid: ValueGrid, offset: Offset) -> ValueGrid:
    return np.roll(grid, offset, axis=(0, 1))


def zoom_valgrid(grid: ValueGrid, factor: RandomInteger) -> ValueGrid:
    return zoom_in(grid, factor)


def zoom_boolgrid(grid: BoolGrid, factor: RandomInteger) -> BoolGrid:
    return zoom_in(grid, factor)


###############################################
# Custom Filtering
###############################################

class RFilters(Enum):
    Modal, Majority = range(2)


def filter_3x3(middle: B, ortho: B, diag: B) -> BoolFilter:
    n_filter = np.zeros((3, 3), dtype=np.uint8)
    if ortho == B.T:
        n_filter = n_filter | ortho_filter
    if diag == B.T:
        n_filter = n_filter | diag_filter
    n_filter[1, 1] = 1 if middle == B.T else 0
    return n_filter


def rfilt(arr: ValueGrid, rfilter: RFilters, s: BoolFilter) -> ValueGrid:
    # if neighbourhood == Structs.Orthogonal:
    #     s = ndimage.generate_binary_structure(2, 1)
    # else:
    #     s = ndimage.generate_binary_structure(2, 2)
    if rfilter == RFilters.Modal:
        return ranks.modal(arr.astype(np.uint8), selem=s)
    else:
        return ranks.majority(arr.astype(np.uint8), selem=s)


def apply_boolf(im: BoolGrid, filt: BoolFilter) -> IntGrid:
    """
    Filters 3x3 patches using the given filter
    """
    return convolve2d(im, filt, mode='same')


def ident_neigh(arr: ValueGrid) -> IntGrid:
    """Outputs the number of identical values in the input for each neighbourhood"""

    def f(patch: np.ndarray):
        # Patch is flattened
        rel_colour = patch[4]

        return np.sum(patch == rel_colour) - 1

    assert arr.shape == (30, 30)
    res = ndimage.generic_filter(
        arr,
        f,
        size=3,
        mode='constant',
        # extra_arguments=
    )
    assert res.shape == (30, 30)
    return res


###############################################
# Misc
###############################################

def tiled(arr: ValueGrid) -> ValueGrid:
    size = (30, 30)
    if len(np.unique(arr)) == 1:
        return arr
    reduced, reducedatt, (l, r, b, t) = ml.reduce_by_attention(arr, arr != 0)
    row_multiplier = size[0] // reduced.shape[0] + 1
    column_multiplier = size[1] // reduced.shape[1] + 1
    return np.tile(reduced, (row_multiplier, column_multiplier))[:size[0], :size[1]]


###############################################
# Output Helpers
###############################################

def obj_to_valgrid(obj: SingleObject) -> ValueGrid:
    return obj[0]


def obj_to_boolgrid(obj: SingleObject) -> BoolGrid:
    return obj[1]


def int_to_real(int_arr: IntGrid) -> RealGrid:
    return int_arr


def bool_to_real(b_arr: BoolGrid) -> RealGrid:
    return b_arr.astype(np.float32)


if __name__ == '__main__':
    sess = lib.Session()
    # test_subject = sess.query(lib.Subject).filter(lib.Subject.name == '228f6490').first()
    test_subject = sess.query(lib.Subject).filter(lib.Subject.name == '1cf80156').first()
    # test_subject = sess.query(lib.Subject).filter(lib.Subject.name == 'd037b0a7').first()
    game = game_from_subject(test_subject)
    pair = game.train_pairs[0]

    fig, (ax1, ax2) = plt.subplots(2, 1)
    pair.visualize(ax1, ax2)
    fig.show()

    # x = lbl_by_bg(pair.get_situation(), structure=Structs.Orthogonal, background=Value.Empty)
    # x = apply_boolf(pair.get_situation() == 1)  # , RFilters.Modal, Structs.Orthogonal)
    # sns.heatmap(x);
    # plt.show()
    # l = coord(pair.get_situation(), 2)
    # l2 = different_cells_in_line(pair.get_situation(), Orientation.horizontal)
    # g = arr_from_line(l, Orientation.vertical)
    # f = filter_3x3(B.F, B.T, B.F)
    labels = lbl_connected(pair.get_situation(), Structs.Full, background=Value.Empty)
    im, mask = object_by_spatial(labels, 0, B.T)
    h = hist(pair.get_situation())
