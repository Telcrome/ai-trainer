import itertools
import random
from typing import Dict, Callable

import trainer.lib
from trainer.ml.data_model import Dataset, Subject


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
    for s_name in ds.get_subject_name_list(split=split):
        yield ds.get_subject_by_name(s_name)


def random_subject_generator(ds: Dataset, split=None):
    subjects = ds.get_subject_name_list(split=split)
    random.shuffle(subjects)
    for s_name in itertools.cycle(subjects):
        te = ds.get_subject_by_name(s_name)
        yield te


def get_img_mask_pair(s: Subject, binary_name: str, struct_name: str, frame_number=-1):
    """
    Returns an (image, mask) pair for a specified frame.

    For a single image no frame has to be specified.
    If no valid frame is specified for a video, an exception is raised.
    If no mask exists, (image, None) is returned
    """
    frame = s.get_binary(binary_name)[frame_number]

    struct_index = list(s.get_binary_model(binary_name)["meta_data"]["structures"].keys()).index(struct_name)

    def mask_condition(binary_model):
        if binary_model['binary_type'] == trainer.lib.BinaryType.ImageMask.value:
            if binary_model['meta_data']['mask_of'] == binary_name:
                if binary_model['meta_data']['frame_number'] == frame_number:
                    return True
        return False

    mask_names = s.get_binary_list_filtered(mask_condition)[0]
    if mask_names:
        mask = s.get_binary(mask_names)[:, :, struct_index]
        return frame, mask
    return frame, None


def random_struct_generator(ds: Dataset, struct_name: str, split=None, verbose=True):
    subjects = ds.get_subject_name_list(split=split)
    N = 0

    # Compute the annotated examples for each subject
    annotations: Dict[str, Dict] = {}
    for s_name in subjects:
        s = ds.get_subject_by_name(s_name)
        s_annos, n_s = s.get_manual_struct_segmentations(struct_name)
        N += n_s
        if s_annos:
            annotations[s_name] = s_annos

    if verbose:
        print(f"{len(subjects)} subjects contain {N} frames with masks")

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
