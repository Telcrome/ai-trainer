import os
from abc import ABC, abstractmethod
from typing import Tuple
import itertools

import numpy as np
import skimage
import sqlalchemy as sa
from sqlalchemy.orm.session import Session

import trainer.lib as lib


def get_test_logits(shape=(50, 50), bounds=(-50, 20)) -> np.ndarray:
    """
    Returns a demo array for testing functionality with logits.

    >>> import trainer.demo_data as dd
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> dd.get_test_logits(shape=(2,))
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


def finite_test_gen(start=0, end=5):
    for item in range(start, end):
        yield item


def infinite_test_gen(first=0):
    for item in itertools.count(first):
        yield item


class DemoDataset(ABC):

    def __init__(self, data_path: str, ds_name: str):
        self.ds_name = ds_name
        self.data_path = data_path
        self.kaggle_storage = os.path.join(self.data_path, 'kaggle_datasets')

    @abstractmethod
    def build_dataset(self, sess: Session = lib.Session()) -> lib.Dataset:
        """
        Builds a new dataset if it does not yet exists in the database.
        """
        if sess is None:
            sess = lib.Session()
        d = sess.query(lib.Dataset).filter(lib.Dataset.name == self.ds_name).first()
        if d is None:
            d = lib.Dataset.build_new(self.ds_name)
        return d
