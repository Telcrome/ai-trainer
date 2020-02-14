"""
Produces (smallish) datasets for testing the functionality of the annotator and machine learning functionality.

Uses artificial data that uses tasks solvable by a human to enable simple demonstration of trainer functionality.

A demo dataset contains the following classes:
- Digit (Modality: MNIST)
- Clothing (Modality: Fashion MNIST)
and the following structures:
-
"""
import os
import random
import tempfile

import numpy as np
import skimage
from torchvision import datasets
from tqdm import tqdm

import trainer.lib as lib


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
        self.mnist_train = datasets.MNIST(os.path.join(storage_path, 'mnist_train'), train=True, download=True)
        self.mnist_test = datasets.MNIST(os.path.join(storage_path, 'mnist_test'), train=False, download=True)

        self.mnist_indices = {}
        self.refill_mnist_indices()

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


def build_mnist(sd: SourceData, max_training=-1) -> lib.Dataset:
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
            x, y = sd.mnist_train.__getitem__(i)
            im_stack = lib.ImStack.build_new(np.asarray(x))
            im_stack.set_class('digit', str(y))
            s.ims.append(im_stack)

            d.sbjts.append(s)
            d.get_split_by_name(split_name).sbjts.append(s)

    append_mnist_split(sd.mnist_train, split_name='train')
    append_mnist_split(sd.mnist_test, split_name='test')
    session.add(d)
    session.commit()
    return d
