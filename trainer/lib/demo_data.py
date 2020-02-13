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


def build_test_subject() -> lib.Subject:
    res = lib.Subject('test_subject')

    im_stack = lib.ImageStack.from_np('astronaut', skimage.data.astronaut())
    im_stack.set_class('occupation', 'astronaut')
    res.add_image_stack(im_stack)

    return res


def build_mnist_subject(src_manager: SourceData, max_digit_ims=(1, 5)) -> lib.Subject:
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
    # astr_im.add_sem_seg(np.zeros((astr_im.get_src().shape[1], astr_im.get_src().shape[2], 2), dtype=np.bool))
    s.add_image_stack(astr_im)

    return s


def build_mnist(data_path: str, sd: SourceData, max_training=-1) -> lib.Dataset:
    """
    Builds an Mnist dataset

    :return: The lib.Dataset
    """
    ds_name = f'mnist{max_training}' if max_training != -1 else f'mnist{60000}'
    if os.path.exists(os.path.join(data_path, ds_name)):
        return lib.Dataset.from_disk(os.path.join(data_path, ds_name))
    d = lib.Dataset.build_new(ds_name, data_path)
    d.stop_auto_save()
    d.add_class('digit', lib.ClassType.Nominal, [str(i) for i in range(10)])

    def append_mnist_split(torch_dataset, split='train'):
        train_n = len(torch_dataset) if max_training == -1 or split != 'train' else max_training
        for i in tqdm(range(train_n)):
            s = lib.Subject(f'subject_{split}_{i}')
            x, y = sd.mnist_train.__getitem__(i)
            im_stack = lib.ImageStack.from_np(f"mnist{i}", np.asarray(x))
            im_stack.set_class('digit', str(y))
            s.add_image_stack(im_stack)
            d.save_subject(s)
            d.append_subject_to_split(s.entity_id, split=split)

    append_mnist_split(sd.mnist_train, split='train')
    append_mnist_split(sd.mnist_test, split='test')
    d.to_disk()
    return d
