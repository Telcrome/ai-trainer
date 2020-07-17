import os
import random

import numpy as np
from torchvision import datasets
from tqdm import tqdm

import trainer.demo_data as dd
import trainer.lib as lib


class MnistDataset(dd.DemoDataset):
    """
    

    >>> import tempfile
    >>> import trainer.demo_data as dd
    >>> dir_path = tempfile.gettempdir()
    >>> mnist_dataset = dd.MnistDataset(dir_path)
    >>> x, y = mnist_dataset.sample_digit(digit=2)
    >>> y
    2
    >>> x.shape
    (28, 28)
    """

    def __init__(self, data_path: str, max_training_examples=-1):
        """
        Builds two splits: mnist train and mnist test
        """
        super().__init__(data_path, 'mnist')
        self.mnist_train = datasets.MNIST(os.path.join(self.data_path, 'mnist_train'), train=True, download=True)
        self.mnist_test = datasets.MNIST(os.path.join(self.data_path, 'mnist_test'), train=False, download=True)
        self.refill_mnist_indices()
        self.mnist_indices = {i: [] for i in range(10)}
        self.n_train = max_training_examples

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

    def build_dataset(self, sess=None) -> lib.Dataset:
        d, sess = super().build_dataset(sess)

        def append_mnist_split(torch_dataset, split_name='train'):
            d.add_split(split_name=split_name)
            train_n = len(torch_dataset) if self.n_train == -1 or split_name != 'train' else self.n_train

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
        sess.add(d)
        sess.commit()
        return d


if __name__ == '__main__':
    lib.reset_complete_database()
    mnist = MnistDataset('D:\\')
    ds = mnist.build_dataset()
