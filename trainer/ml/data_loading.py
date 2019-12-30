import itertools
import os
import random
from typing import Dict, Callable

from trainer.ml import Dataset, Subject


def iterate_over_samples(ds: Dataset, f: Callable[[Subject], Subject]):
    """
    Applies a function on every subject in this dataset.
    :param ds: The dataset to be modified
    :param f: f takes a subject and can modify it. The result is automatically saved
    :return:
    """
    for s_name in ds.get_subject_name_list():
        s = ds.get_subject_by_name(s_name)
        s = f(s)
        s.to_disk()


def get_subject_gen(ds: Dataset, split: str = None):
    """
    Iterates once through the dataset. Intended for custom exporting, not machine learning.
    """
    if split is None:
        subjects = ds._json_model["subjects"]
    else:
        subjects = ds._json_model["splits"][split]

    for s_name in subjects:
        yield ds.get_subject_by_name(s_name)


def random_subject_generator(ds: Dataset, split=None):
    if split is None:
        te_names = ds._json_model["subjects"]
    else:
        te_names = ds._json_model["splits"][split]
    random.shuffle(te_names)

    for s_name in itertools.cycle(te_names):
        te = ds.get_subject_by_name(s_name)
        yield te


def random_struct_generator(ds: Dataset, struct_name: str, split=None):
    if split is None:
        subjects = ds._json_model["subjects"]
    else:
        subjects = ds._json_model["splits"][split]

    # Compute the annotated examples for each subject
    annotations: Dict[str, Dict] = {}
    for s_name in subjects:
        s = ds.get_subject_by_name(s_name)
        s_annos = s.get_manual_struct_segmentations(struct_name)
        if s_annos:
            annotations[s_name] = s_annos
    print(annotations)

    for s_name in itertools.cycle(annotations.keys()):
        s = ds.get_subject_by_name(s_name)
        # Randomly pick the frame that will be trained with
        a = annotations[s_name]
        b_name = random.choice(list(annotations[s_name].keys()))
        m_name = random.choice(annotations[s_name][b_name])
        # Build the training example with context
        struct_index = list(s.get_binary_model(b_name)["meta_data"]["structures"].keys()).index(struct_name)
        yield s.get_binary(b_name), s.get_binary(m_name)[:, :, struct_index], \
              s.get_binary_model(m_name)["meta_data"]['frame_number']
