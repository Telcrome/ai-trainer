"""
Loading data from the data model (Dataset and Subject) exists as a separate module,
because there are so many different, complex things to be done which would bloat the respective classes.
"""

import random
from typing import Union

import trainer.lib as lib
from trainer.lib import Subject


def image_stack_classification_preprocessor(s: Subject, class_name: str) -> Union[lib.ImageStack, None]:
    """
    Pulls an imagestack with the class in question from the subject.

    >>> import trainer.lib.demo_data as demo_data
    >>> s = demo_data.build_test_subject()
    >>> im_stack = image_stack_classification_preprocessor(s, 'occupation')
    >>> im_stack.entity_id
    'astronaut'

    :param s: The lib.Subject
    :param class_name: One imagestack with this class will be returned
    :return: The lib.ImageStack. If there is no Imagestack with this class, return None
    """
    im_stacks = [s.get_image_stack(k) for k in s.get_image_stack_keys()]
    if im_stacks:
        return random.choice(list(filter(lambda im_stack: im_stack.contains_class(class_name), im_stacks)))
    else:
        return None
