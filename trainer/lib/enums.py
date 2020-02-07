"""
A module for collecting the enumerations that contain the hardcoded values
"""

from enum import Enum


class MaskType(Enum):
    """
    Possible types that a mask can have.

    - blob: straightforward region. Is used for most segmentation tasks
    - A point is usually segmented as a small circle and then postprocessed to be the center of that circle
    - A line is usually segmented as a sausage and then postprocessed to a single response-line
    """
    Unknown = 'unknown'
    Blob = 'blob'
    Point = 'point'
    Line = 'line'


class BinarySaveProvider(Enum):
    Pickle = 0
    Numpy = 1


class BinaryType(Enum):
    """
    Multiple different types of binaries are supported.

    Image stacks are used for images, videos and 3D images.
    Shape of an image stack: [#frames, width, height, #channels]

    Segmentation Masks ('img_mask') are used to store every annotated structure for one frame of an imagestack.
    Shape of a mask: [width, height, #structures]

    Miscellaneous objects are general pickled objects.
    """

    @staticmethod
    def provider_map():
        return {
            BinaryType.Unknown.value: BinarySaveProvider.Pickle,
            BinaryType.NumpyArray.value: BinarySaveProvider.Numpy,
            BinaryType.ImageStack.value: BinarySaveProvider.Numpy,
            BinaryType.ImageMask.value: BinarySaveProvider.Numpy,
            BinaryType.TorchStateDict.value: BinarySaveProvider.Pickle
        }

    Unknown = 'unknown'
    NumpyArray = 'numpy_array'
    ImageStack = 'image_stack'
    ImageMask = 'img_mask'
    TorchStateDict = 'torch_state_dict'


class ClassType(Enum):
    Binary = 'binary'
    Nominal = 'nominal'
    Ordinal = 'ordinal'


class ClassSelectionLevel(Enum):
    SubjectLevel = "Subject Level"
    BinaryLevel = "Binary Level"
    FrameLevel = "Frame Level"
