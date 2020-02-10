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
        self.mnist = datasets.MNIST(os.path.join(storage_path, 'mnist'), train=True, download=True)

        self.mnist_indices = {}
        self.refill_mnist_indices()

    def refill_mnist_indices(self):
        self.mnist_indices = {i: [] for i in range(10)}
        for i in range(len(self.mnist)):
            _, y = self.mnist.__getitem__(i)
            self.mnist_indices[y].append(i)

    def sample_digit(self, digit=0):
        if len(self.mnist_indices[digit]) == 0:
            self.refill_mnist_indices()
        index = random.choice(self.mnist_indices[digit])
        self.mnist_indices[digit].remove(index)
        x, y = self.mnist.__getitem__(index)
        x = np.asarray(x)
        return x, y


def get_test_logits(shape=(50, 50), bounds=(-50, 20)) -> np.ndarray:
    """
    Returns a demo array for testing functionality with logits.

    >>> import trainer.lib as lib
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> lib.get_test_logits(shape=(2,))
    array([-5.28481063, -2.39723662])

    :param shape: Shape of the test data. For one-dimensional data use (w, ).
    :param bounds: Optional to specify the ceiling and floor of the output using a 2-Tuple (floor, ceiling)
    :return: Demo logits
    """
    low, high = bounds
    return np.random.randint(low=low, high=high, size=shape) + np.random.rand(*shape)


def build_random_subject(d: lib.Dataset, src_manager: SourceData, max_digit_ims=(1, 5)) -> lib.Subject:
    """
    Samples a random subject.
    """
    digit_class = random.randint(0, 9)
    s = lib.Subject(lib.create_identifier('subject'))
    s.set_class('digit', str(digit_class))

    # digit classification
    for i in range(random.randint(*max_digit_ims)):
        x, y = src_manager.sample_digit(digit=digit_class)
        im_stack = lib.ImageStack.from_np(lib.create_identifier(f"mnist{i}"), x)
        im_stack.set_class('digit', str(digit_class))
        s.add_image_stack(im_stack)

    astr_im = lib.ImageStack.from_np('astronaut', skimage.data.astronaut())
    s.add_image_stack(astr_im)

    return s
