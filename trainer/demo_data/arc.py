"""
Provides utilities and the ARC-dataset in the trainer-format.

See https://github.com/fchollet/ARC for details.
"""
from __future__ import annotations
from functools import reduce
import json
import os
from typing import List, Dict, Any
from enum import Enum

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import trainer.demo_data as dd
import trainer.lib as lib
import trainer.ml as ml


def array_from_json(list_repr: List[List], depth=10) -> np.ndarray:
    width, height = len(list_repr), len(list_repr[0])
    res = np.zeros((width, height, depth), dtype=np.uint8)
    for x in range(width):
        for y in range(height):
            depth_val = list_repr[x][y]
            res[x, y, depth_val] = 1
    return res


class ArcDataset(dd.DemoDataset):

    def __init__(self, data_path: str):
        super().__init__(data_path, 'arc')
        self.arc_path = os.path.join(self.kaggle_storage, 'abstraction-and-reasoning-challenge')
        if not os.path.exists(self.arc_path):
            raise FileNotFoundError("The files required for building the arc dataset could not be found")

    def create_arc_split(self, d: lib.Dataset, ss_tpl: lib.SemSegTpl, split_name='training'):
        d.add_split(split_name=split_name)
        p = os.path.join(self.arc_path, split_name)
        for file_path in tqdm(os.listdir(p), desc=split_name):
            f_name = os.path.split(file_path)[-1]
            s_name = os.path.splitext(f_name)[0]
            s = lib.Subject.build_new(s_name)
            with open(os.path.join(p, f_name), 'r') as f:
                json_content = json.load(f)
            for key in json_content:
                for maze in json_content[key]:
                    extra_info = {'purpose': key}
                    # Input Image
                    input_json = maze['input']
                    input_im = array_from_json(input_json, depth=10)
                    im_stack = lib.ImStack.build_new(src_im=input_im, extra_info=extra_info)

                    # If the solution is given, add:
                    if 'output' in maze:
                        output_json = maze['output']
                        gt_arr = array_from_json(output_json, depth=10).astype(np.bool)
                        im_stack.add_ss_mask(
                            gt_arr,
                            sem_seg_tpl=ss_tpl,
                            ignore_shape_mismatch=True)

                        # Add metadata
                        im_stack.extra_info['sizeeq'] = (input_im.shape == gt_arr.shape)

                    s.ims.append(im_stack)
            s.extra_info['all_have_target'] = reduce(lambda x, y: x and y,
                                                     ['sizeeq' in im.extra_info for im in s.ims])
            if s.extra_info['all_have_target']:
                s.extra_info['sizeeq'] = reduce(lambda x, y: x and y,
                                                [im.extra_info['sizeeq'] for im in s.ims])
            d.get_split_by_name(split_name).sbjts.append(s)

    def build_dataset(self, sess=lib.Session()) -> lib.Dataset:
        d = super().build_dataset(sess=sess)

        # Dataset does not exist yet, build it!
        ss_tpl = lib.SemSegTpl.build_new(
            'arc_colors',
            {
                "Zero": lib.MaskType.Blob,
                "One": lib.MaskType.Blob,
                "Two": lib.MaskType.Blob,
                "Three": lib.MaskType.Blob,
                "Four": lib.MaskType.Blob,
                "Five": lib.MaskType.Blob,
                "Six": lib.MaskType.Blob,
                "Seven": lib.MaskType.Blob,
                "Eight": lib.MaskType.Blob,
                "Nine": lib.MaskType.Blob,
            }
        )
        sess.add(ss_tpl)

        self.create_arc_split(d, ss_tpl, 'training')
        # self.create_arc_split(d, ss_tpl, 'test')
        self.create_arc_split(d, ss_tpl, 'evaluation')
        sess.add(d)
        sess.commit()
        return d


class Value(Enum):

    @staticmethod
    def from_ind(c_ind: int) -> Value:
        if c_ind in c_value:
            return c_value[c_ind]
        else:
            return Value.Empty

    Empty, Black, Blue, Red, Green, Yellow, Grey, Pink, Orange, Cyan, Magenta = range(11)


c_value = lib.make_converter_dict_for_enum(Value)


class Pair:

    def __init__(self, situation: np.ndarray, target: np.ndarray):
        self._situation = situation
        self._target = target

    def get_situation(self):
        return ml.pad(ml.one_hot_to_cont(self._situation) + 1)

    def get_target(self):
        return ml.pad(ml.one_hot_to_cont(self._target) + 1)

    def visualize(self, ax1, ax2):
        ax1.set_title("Initial Situation")
        sns.heatmap(ml.one_hot_to_cont(self._situation), ax=ax1, vmin=0, vmax=10)
        ax2.set_title("Target Situation")
        sns.heatmap(ml.one_hot_to_cont(self._target), ax=ax2, vmin=0, vmax=10)


def pair_from_imstack(im: lib.ImStack) -> Pair:
    x, y = im.get_ndarray()[0, :, :, :], im.semseg_masks[0].get_ndarray()[:, :, :]
    return Pair(np.rollaxis(x, 2, 0), np.rollaxis(y, 2, 0).astype(np.uint8))


def extract_train_test(s: lib.Subject):
    train_examples: List[lib.ImStack] = []
    test_examples: List[lib.ImStack] = []
    # noinspection PyTypeChecker
    for imstack in s.ims:
        if imstack.extra_info['purpose'] == 'train':
            train_examples.append(imstack)
        else:
            test_examples.append(imstack)
    assert (len(train_examples) > 0 and len(test_examples) > 0), f"Something is wrong with {s.name}"
    return train_examples, test_examples


def encode_depthmap(x: np.ndarray, n_classes=11, max_grid=30) -> np.ndarray:
    inp = np.zeros((n_classes, max_grid, max_grid), dtype=np.float32)
    inp[0, x.shape[0]:, :] = 1.
    inp[0, :, x.shape[1]:] = 1.
    x_rolled = np.rollaxis(x, 2, 0)
    inp[1:, :x_rolled.shape[1], :x_rolled.shape[2]] = x_rolled
    return inp


def plot_as_heatmap(arc_field: np.ndarray, ax=None, title='title') -> None:
    if arc_field.dtype == np.bool and np.max(arc_field) < 2:
        sns.heatmap(arc_field, ax=ax, vmin=0, vmax=1)
    else:
        sns.heatmap(arc_field, ax=ax, vmin=0, vmax=11)
    if ax is None:
        plt.title(title)
        plt.show()


PAIR_STATE = Dict[str, Any]


def plot_pairs(pairs: List[Pair], title='training', magnification=3) -> plt.Figure:
    fig, axs = plt.subplots(pairs.__len__(), 2,
                            figsize=(magnification, (magnification // 2) * pairs.__len__()))
    assert isinstance(axs, np.ndarray)
    fig.suptitle(title)
    for i, train_pair in enumerate(pairs):
        if len(axs.shape) > 1:
            ax1, ax2 = axs[i, 0], axs[i, 1]
        else:
            ax1, ax2 = axs
        sns.heatmap(train_pair.get_situation(), ax=ax1, vmin=0, vmax=11)
        sns.heatmap(train_pair.get_target(), ax=ax2, vmin=0, vmax=11)
    return fig


def plot_game(g: Game) -> Tuple[plt.Figure, plt.Figure]:
    return plot_pairs(g.train_pairs, title="Training"), plot_pairs(g.test_pairs, title="Test")


def game_from_subject(s: lib.Subject, logging=True) -> Game:
    if logging:
        lib.logger.debug_var(f"Extracting a game from subject {s.name}")
    train_examples, test_examples = extract_train_test(s)

    g = Game(
        [pair_from_imstack(im) for im in train_examples],
        [pair_from_imstack(im) for im in test_examples],
    )
    return g


def extract_objts(grid: np.ndarray, indices: List[np.ndarray]) -> List[np.ndarray]:
    """
    Given an ARC field and
    """
    res = []
    for o in indices:
        a = np.zeros_like(grid)
        a[o] = grid[o]
        res.append(a)
    return res


class Game:

    def __init__(self,
                 train_pairs: List[Pair],
                 test_pairs: List[Pair]):
        self.train_pairs = train_pairs
        self.test_pairs = test_pairs

    @classmethod
    def extract_from_subject(cls, s: lib.Subject):
        lib.logger.debug_var(f"Extracting a game from subject {s.name}")

        train_examples, test_examples = extract_train_test(s)

        def extract_pair(im: lib.ImStack) -> Pair:
            x, y = im.get_ndarray()[0, :, :, :], im.semseg_masks[0].get_ndarray()[:, :, :]
            return Pair(np.rollaxis(x, 2, 0), np.rollaxis(y, 2, 0).astype(np.uint8))

        g = cls(
            [extract_pair(im) for im in train_examples],
            [extract_pair(im) for im in test_examples]
        )
        return g

    def get_train_colors(self):
        # res = []  # Not necessary, because situation and target is padded by 0 now anyways
        uniques = []
        for train_pair in self.train_pairs:
            uniques.append(np.unique(train_pair.get_situation(), return_counts=True))
            uniques.append(np.unique(train_pair.get_target(), return_counts=True))

        counts = np.zeros(11, dtype=np.int)
        for num_name, c in uniques:
            for i, num_id in enumerate(num_name):
                counts[num_id] += c[i]
        c_sort = np.flip(np.argsort(counts))
        res = list(c_sort)
        return res

    # def get_colour_counts(self) -> np.ndarray:
    #     res = np.zeros(10)
    #     for train_pair in self.train_pairs:
    #         # Do not count zero, because the empty space has another semantic meaning than the other values
    #         sit_colours = np.unique(train_pair.get_situation()[train_pair.get_situation() != 0], return_counts=True)
    #         for i, colour in enumerate(sit_colours[0]):
    #             res[colour - 1] += sit_colours[1][i]
    #     return res

    def compute_boxes(self):
        boxes = []

        # If all targets have the same shape, add this box:
        def unpad(padded: np.array) -> np.ndarray:
            pure_target, att, _ = ml.reduce_by_attention(padded, padded != 0)
            return pure_target

        target_shape = unpad(self.train_pairs[0].get_target()).shape

        for pair in self.train_pairs:
            if target_shape != unpad(pair.get_target()).shape:
                target_shape = None
                break
        if target_shape is not None:
            boxes.append(target_shape)

        arrs = []
        for box_shape in boxes:
            arr = np.zeros((30, 30), dtype=np.bool)
            arr[:box_shape[0], :box_shape[1]] = True
            arrs.append(arr)
        return arrs

    def get_mults(self):
        # Currently used for computing big offsets and as a zoom parameter
        res = [2, 3, 4, 5]
        return res

    def get_states(self) -> Tuple[List[PAIR_STATE], List[PAIR_STATE]]:
        count_index_names = {
            0: 'First',
            1: 'Second',
            2: 'Third',
            3: 'Fourth',
            4: 'Fifth',
            5: 'Sixth',
            6: 'Seventh',
            7: 'Eighth',
            8: 'Ninth',
            9: 'Tenth',
            10: 'Eleventh'
        }

        def pair_to_state(pair: Pair) -> PAIR_STATE:
            return {
                'grid': pair.get_situation(),
                'target': pair.get_target(),
                'boxes': self.compute_boxes(),
                'values': self.get_train_colors(),
                # 'game_counts': self.get_colour_counts(),
                'index_name_conv': count_index_names,
                'train_targets': [t.get_target() for t in self.train_pairs],
                'special_locations': [(0, 0)],
                'mults': self.get_mults()
            }

        train_states = [pair_to_state(p) for p in self.train_pairs]
        test_states = [pair_to_state(p) for p in self.test_pairs]
        return train_states, test_states


if __name__ == '__main__':
    lib.reset_complete_database()
    arc = ArcDataset('D:\\')
    ds = arc.build_dataset()
