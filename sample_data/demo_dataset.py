"""
Produces (smallish) datasets for testing the functionality of the annotator and machine learning functionality.
"""

import os
import random

import numpy as np
import tensorflow as tf
import PySimpleGUI as sg

from trainer.ml import Subject, Dataset, ClassType
from trainer.bib import load_grayscale_from_disk, standalone_foldergrab, create_identifier
from trainer.bib import MaskType

US_BONE_DATASET = ("https://rwth-aachen.sciebo.de/s/1qO95mdEjhoUBMf/download", "crucial_ligament_diagnosis")


def build_example_te(name: str) -> Subject:
    test_im_path = "./example_image.png"
    test_gt_path = "./example_gt.png"
    te = Subject.build_empty(name)

    im = load_grayscale_from_disk(test_im_path)
    te.add_source_image_by_arr(im, binary_name="src", structures={"bone": MaskType.Line})

    gt = load_grayscale_from_disk(test_gt_path).astype(np.bool)
    te.add_new_gt_by_arr(gt, mask_of="src", structure_names=["bone"])

    from skimage.data import astronaut
    te.add_source_image_by_arr(src_im=astronaut(), binary_name="astronaut_image")

    return te


def create_mnist(p: str, n: str = 'mnist', n_patients=10):
    """
    Mimics a patient database by solving a classification issue with 10 classes and multiple images per patient.

    Each patient has one of the classes
    - ZERO
    - ONE
    - ...
    - NINE
    :return:
    """
    d = Dataset.build_new(n, p)

    fashion_mnist = tf.keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['TShirtOrTop', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'AnkleBoot']

    class_name = "diagnosis"
    d.add_class(class_name, ClassType.Nominal, values=class_names)

    def get_sample_of_class(class_value: str):
        label_indices = np.argwhere(train_labels == class_names.index(class_value))
        sample_index = label_indices[random.randint(0, label_indices.shape[0]), 0]
        return train_images[sample_index, :, :]

    for p_i in range(n_patients):
        sbjct = Subject.build_empty(f"patient{p_i}")
        class_sample = random.choice(class_names)
        for i_i in range(random.randint(1, 3)):
            one_im = get_sample_of_class(class_sample)
            seg_structs = {'outline': MaskType.Line.value,
                           'sleeve': MaskType.Blob.value}
            b_name = create_identifier(hint=f'image{i_i}')
            sbjct.add_source_image_by_arr(one_im, b_name, structures=seg_structs)
            sbjct.set_class(class_name, class_sample, for_dataset=d, for_binary=b_name)
        sbjct.set_class(class_name, class_sample, for_dataset=d)
        sg.OneLineProgressMeter('Creating patient', p_i + 1, n_patients, 'key',
                                f'Subject: {sbjct.name} of class {class_sample}')
        d.save_subject(sbjct)

    d.to_disk()


if __name__ == '__main__':
    parent_path, input_keys = standalone_foldergrab(
        folder_not_file=True,
        title="Select the parent folder",
        optional_inputs=[
            ("Name of the dataset", "d_name"),
            ("Max Number Subjects", "#s")
        ],
        optional_choices=[
            ("Dataset Type", "ds_type", ["Fashion Mnist", "Ultrasound"])
        ]
    )
    dataset_name = input_keys['d_name']
    N = input_keys['#s']
    d_path = os.path.join(parent_path, dataset_name)
    if os.path.exists(d_path):
        Dataset.from_disk(d_path).delete_on_disk()
    if input_keys['ds_type'] == 'Fashion Mnist':
        create_mnist(parent_path, dataset_name, n_patients=int(N))
    elif input_keys['ds_type'] == 'Ultrasound':
        print('Create Ultrasound')
        raise NotImplementedError()
