from enum import Enum


class MaskType(Enum):
    """
    Possible types that a mask can have.
    Use blob if
    """
    Unknown = 'unknown'
    Blob = 'blob'
    Point = 'point'
    Line = 'line'


class BinaryType(Enum):
    """
    Two different types of binaries are supported.
    Image stacks are used for images, videos and 3D images.
    Shape of an image stack: [#frames, width, height, #channels]

    Masks are used to store every annotated structure for one frame of an imagestack.
    Shape of a mask: [width, height, #structures]
    """
    Unknown = 'unknown'
    ImageStack = 'image_stack'
    ImageMask = 'img_mask'


class ClassType(Enum):
    Binary = 'binary'
    Nominal = 'nominal'
    Ordinal = 'ordinal'


class ClassSelectionLevel(Enum):
    SubjectLevel = "Subject Level"
    BinaryLevel = "Binary Level"
    FrameLevel = "Frame Level"
