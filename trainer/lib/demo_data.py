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


def get_dummy_jsonclass(jc_name="Test Json Class"):
    """
    Intended to be used for testing functionality concerned with the basic Jsonclass.

    >>> import trainer.lib as lib
    >>> jc = lib.get_dummy_jsonclass()
    >>> jc.name
    'Test Json Class'
    >>> jc.get_binary('b1')  # A small array is contained in the example
    array([1, 2, 3])
    >>> jc.get_binary('obj')
    {'this': 'is', 'an': 'object'}

    :param jc_name: Name of the Jsonclass
    :return: A reference to a Jsonclass
    """
    dir_path = tempfile.gettempdir()

    res = lib.JsonClass(jc_name, {
        'Attribute 1': "Value 1"
    })
    res.to_disk(dir_path)
    res.add_binary('b1', np.array([1, 2, 3]), b_type=lib.BinaryType.NumpyArray.value)

    res.add_binary('picture', skimage.data.retina(), b_type=lib.BinaryType.ImageStack.value)

    python_obj = {
        "this": "is",
        "an": "object"
    }
    res.add_binary('obj', python_obj, lib.BinaryType.Unknown.value)

    return res


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


class SourceData:

    def __init__(self, storage_path: str):
        self.storage_folder = storage_path
        self.kaggle_storage = os.path.join(storage_path, 'kaggle_datasets')
        self.mnist = datasets.MNIST(data_folder, train=True, download=True)

        self.mnist_indices = {i: [] for i in range(10)}
        for i in range(len(self.mnist)):
            _, y = self.mnist.__getitem__(i)
            self.mnist_indices[y].append(i)

    def sample_digit(self, digit=0):
        index = random.choice(self.mnist_indices[digit])
        del self.mnist_indices[index]  # Make sure this very example is not used again
        return self.mnist.__getitem__(index)


def build_random_subject(d: lib.Dataset, src_manager: SourceData, max_digit_ims=5) -> lib.Subject:
    """
    Samples a random subject.
    """
    s = lib.Subject.build_empty(lib.create_identifier())

    digit_class = random.randint(0, 9)

    # digit classification
    for i in range(random.randint(1, max_digit_ims)):
        x, y = src_manager.sample_digit(digit=digit_class)
        s.add_source_image_by_arr(x, binary_name=lib.create_identifier(f"mnist{i}"))
        s.set_class('digit', str(digit_class), for_dataset=d)

    s.add_source_image_by_arr(src_im=skimage.data.astronaut(), binary_name="astronaut_image")

    return s


def create_dataset(dir_path: str, ds_name: str = 'demo', n_subjects=10):
    """
    Mimics the demonstration dataset. See module docstring for details.=
    """
    src_data = SourceData(dir_path)
    d = lib.Dataset.build_new(ds_name, dir_path)

    for _ in range(n_subjects):
        s = build_random_subject(d, src_manager=src_data)
        d.save_subject(s)

    return d
    # fashion_mnist = tf.keras.datasets.fashion_mnist
    #
    # (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    # class_names = ['TShirtOrTop', 'Trouser', 'Pullover', 'Dress', 'Coat',
    #                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'AnkleBoot']
    #
    # class_name = "fashion_type"
    # d.add_class(class_name, ClassType.Nominal, values=class_names)
    #
    # def get_sample_of_class(class_value: str):
    #     label_indices = np.argwhere(train_labels == class_names.index(class_value))
    #     sample_index = label_indices[random.randint(0, label_indices.shape[0]), 0]
    #     return train_images[sample_index, :, :]
    #
    # for p_i in range(n_subjects):
    #     sbjct = Subject.build_empty(f"patient{p_i}")
    #     d.save_subject(sbjct)
    #     class_sample = random.choice(class_names)
    #     for i_i in range(random.randint(1, 3)):
    #         one_im = get_sample_of_class(class_sample)
    #         seg_structs = {'outline': MaskType.Line.value,
    #                        'sleeve': MaskType.Blob.value}
    #         b_name = create_identifier(hint=f'image{i_i}')
    #         sbjct.add_source_image_by_arr(one_im, b_name, structures=seg_structs)
    #         sbjct.set_class(class_name, class_sample, for_dataset=d, for_binary=b_name)
    #     sbjct.set_class(class_name, class_sample, for_dataset=d)
    #     sg.OneLineProgressMeter('Creating patient', p_i + 1, n_subjects, 'key',
    #                             f'Subject: {sbjct.name} of class {class_sample}')
    #     sbjct.to_disk()
    # d.to_disk()


if __name__ == '__main__':
    # Enter your path for storing the standard machine learning datasets:
    data_folder = 'D:/'
    print(f"Downloading source data into {data_folder}")
