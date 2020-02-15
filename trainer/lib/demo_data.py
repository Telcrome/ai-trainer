"""
Produces (smallish) datasets for testing the functionality of the annotator and machine learning functionality.

Uses artificial data that uses tasks solvable by a human to enable simple demonstration of trainer functionality.

A demo dataset contains the following classes:
- Digit (Modality: MNIST)
- Clothing (Modality: Fashion MNIST)
and the following structures:
-
"""
import json
import os
import random
import tempfile
from typing import List, Union

import numpy as np
import skimage
from torchvision import datasets
from tqdm import tqdm

import trainer.lib as lib


def array_from_json(list_repr: List[List], depth=10) -> np.ndarray:
    width, height = len(list_repr), len(list_repr[0])
    res = np.zeros((width, height, depth), dtype=np.uint8)
    for x in range(width):
        for y in range(height):
            depth_val = list_repr[x][y]
            res[x, y, depth_val] = 1
    return res


class SourceData:
    """
    Intended to be used to load standard machine learning datasets as mock-up data for the trainer dataset format.

    >>> import tempfile
    >>> from trainer.lib.demo_data import SourceData
    >>> dir_path = tempfile.gettempdir()
    >>> sd = SourceData(dir_path)
    >>> x, y = sd.sample_digit(digit=2)
    >>> y
    2
    >>> x.shape
    (28, 28)
    """

    def __init__(self, storage_path: str):
        self.storage_folder = storage_path
        self.kaggle_storage = os.path.join(storage_path, 'kaggle_datasets')
        self.arc_path = os.path.join(self.kaggle_storage, 'abstraction-and-reasoning-challenge')

        self.mnist_train = datasets.MNIST(os.path.join(storage_path, 'mnist_train'), train=True, download=True)
        self.mnist_test = datasets.MNIST(os.path.join(storage_path, 'mnist_test'), train=False, download=True)
        self.refill_mnist_indices()
        self.mnist_indices = {}

        self.arc_dataset: Union[None, lib.Dataset] = None

    def build_arc(self, sess):
        if os.path.exists(self.arc_path):
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

            self.arc_dataset = lib.Dataset.build_new('arc')
            self.create_arc_split(ss_tpl, 'training')
            self.create_arc_split(ss_tpl, 'test')
            self.create_arc_split(ss_tpl, 'evaluation')
            sess.add(self.arc_dataset)
            sess.commit()
            return self.arc_dataset
        return None

    def refill_mnist_indices(self):
        self.mnist_indices = {i: [] for i in range(10)}
        for i in range(len(self.mnist_train)):
            _, y = self.mnist_train.__getitem__(i)
            self.mnist_indices[y].append(i)

    def sample_digit(self, digit=0):
        if len(self.mnist_indices[digit]) == 0:
            self.refill_mnist_indices()
        index = random.choice(self.mnist_indices[digit])
        self.mnist_indices[digit].remove(index)
        x, y = self.mnist_train.__getitem__(index)
        x = np.asarray(x)
        return x, y

    def build_mnist(self, max_training=-1) -> lib.Dataset:
        """
        Builds two splits: mnist train and mnist test

        :return: The lib.Dataset
        """
        session = lib.Session()
        d = lib.Dataset.build_new('mnist')

        def append_mnist_split(torch_dataset, split_name='train'):
            d.add_split(split_name=split_name)
            train_n = len(torch_dataset) if max_training == -1 or split_name != 'train' else max_training

            for i in tqdm(range(train_n)):  # Build the subjects of this split and append them to the split
                s = lib.Subject.build_new(f'subject_{split_name}_{i}')
                x, y = self.mnist_train.__getitem__(i)
                im_stack = lib.ImStack.build_new(np.asarray(x))
                im_stack.set_class('digit', str(y))
                s.ims.append(im_stack)

                d.sbjts.append(s)
                d.get_split_by_name(split_name).sbjts.append(s)

        append_mnist_split(self.mnist_train, split_name='train')
        append_mnist_split(self.mnist_test, split_name='test')
        session.add(d)
        session.commit()
        return d

    def create_arc_split(self, ss_tpl: lib.SemSegTpl, split_name='training'):
        res = self.arc_dataset.add_split(split_name=split_name)
        p = os.path.join(self.arc_path, split_name)
        for file_path in tqdm(os.listdir(p)):
            f_name = os.path.split(file_path)[-1]
            s_name = os.path.splitext(f_name)[0]
            s = lib.Subject.build_new(s_name)
            with open(os.path.join(p, f_name), 'r') as f:
                json_content = json.load(f)
            for key in json_content:
                extra_info = {'purpose': key}
                for maze in json_content[key]:
                    # Input Image
                    input_json = maze['input']
                    input_im = array_from_json(input_json, depth=10)
                    im_stack = lib.ImStack.build_new(src_im=input_im, extra_info=extra_info)

                    # If the solution is given, add:
                    if 'output' in maze:
                        output_json = maze['output']
                        im_stack.add_ss_mask(
                            array_from_json(output_json, depth=10).astype(np.bool),
                            sem_seg_tpl=ss_tpl,
                            ignore_shape_mismatch=True)

                    s.ims.append(im_stack)
            self.arc_dataset.get_split_by_name(split_name).sbjts.append(s)


def get_test_logits(shape=(50, 50), bounds=(-50, 20)) -> np.ndarray:
    """
    Returns a demo array for testing functionality with logits.

    >>> from trainer.lib.demo_data import get_test_logits
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> get_test_logits(shape=(2,))
    array([-5.28481063, -2.39723662])

    :param shape: Shape of the test data. For one-dimensional data use (w, ).
    :param bounds: Optional to specify the ceiling and floor of the output using a 2-Tuple (floor, ceiling)
    :return: Demo logits
    """
    low, high = bounds
    return np.random.randint(low=low, high=high, size=shape) + np.random.rand(*shape)


def build_test_subject() -> lib.Subject:
    res = lib.Subject.build_new('test_subject')

    im_stack = lib.ImStack.build_new(skimage.data.astronaut())
    im_stack.set_class('occupation', 'astronaut')
    res.add_image_stack(im_stack)

    return res
