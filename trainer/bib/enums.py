from enum import Enum


class MaskType(Enum):
    Unknown = 'unknown'
    Blob = 'blob'
    Point = 'point'
    Line = 'line'


class BinaryType(Enum):
    Unknown = 'unknown'
    ImageStack = 'image_stack'  # Includes videos and volumetric data, shape: (frames, width, height, channels)
    ImageMask = 'img_mask'  # One mask per frame, shape: (width, height, structures)


class ClassType(Enum):
    Binary = 'binary'
    Nominal = 'nominal'
    Ordinal = 'ordinal'


class ClassSelectionLevel(Enum):
    SubjectLevel = "Subject Level"
    BinaryLevel = "Binary Level"
    FrameLevel = "Frame Level"
