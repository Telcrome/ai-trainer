"""
Data Model
----------

The data model aims to simplify machine learning on complex data structures.
For example, classifying a subject (medical patient) by both its gender and between 1 and 4 ultrasound videos.

A dataset contains:

- Subjects (Which are the training examples)
- Model Weights
- Json config files
  - Class definitions
  - Segmentation mask definitions

A Subject contains:

- Images & masks
- Classes
- Json config files

Multiple different types of binaries are supported.

Image stacks are used for images, videos and 3D images.
Shape of an image stack: [#frames, width, height, #channels]

Segmentation Masks ('img_mask') are used to store every annotated structure for one frame of an imagestack.
Shape of a mask: [width, height, #structures]

Miscellaneous objects are general pickled objects.
"""

from __future__ import annotations  # Important for function annotations of symbols that are not loaded yet

from ast import literal_eval as make_tuple
from enum import Enum
from typing import List, Dict

import numpy as np
import sqlalchemy as sa
import sqlalchemy.dialects.postgresql as pg
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# engine = create_engine('sqlite:///:memory:', echo=True)
con_string = 'postgresql+psycopg2://postgres:!supi1324!@127.0.0.1:5432/test4'
engine = create_engine(con_string)
# engine = create_engine('sqlite:///./test.db')
Session = sessionmaker(bind=engine)

Base = declarative_base()

TABLENAME_CLASSVALUES = 'classvalues'
TABLENAME_CLASSDEFINITIONS = 'classdefinitions'
TABLENAME_CLASSIFICATIONS = 'classifications'
TABLENAME_SEMSEGCLASS = 'semsegtclasses'
TABLENAME_SEMSEGTPL = 'semsegtpls'
TABLENAME_SEM_SEG = 'semsegmasks'
TABLENAME_IM_STACKS = 'imagestacks'
TABLENAME_SUBJECTS = 'subjects'
TABLENAME_SPLITS = 'splits'
TABLENAME_DATASETS = 'datasets'


class NumpyBinary:
    binary = sa.Column(sa.LargeBinary)
    shape = sa.Column(sa.String())
    dtype = sa.Column(sa.String())

    def set_array(self, arr: np.ndarray) -> None:
        self.binary = arr.tobytes()
        self.shape = str(arr.shape)[1:-1]
        self.dtype = str(arr.dtype)

    def get_ndarray(self) -> np.ndarray:
        return np.frombuffer(self.binary, dtype=self.dtype).reshape(make_tuple(f'({self.shape})'))

    def __repr__(self):
        return f"Numpy Binary with shape ({self.shape}) and type {self.dtype}>"


class ClassType(Enum):
    Binary = 'binary'
    Nominal = 'nominal'
    Ordinal = 'ordinal'


class Classifiable:
    classes = sa.Column(pg.JSONB())

    def set_class(self, class_name: str, class_val: str):
        if self.classes:
            self.classes[class_name] = class_val
        else:
            self.classes = {class_name: class_val}

    def remove_class(self, class_name: str):
        self.classes.pop(class_name)

    def get_class(self, class_name: str):
        return self.classes[class_name]

    @classmethod
    def query_all_with_class(cls, session: sa.orm.session.Session, class_name: str):
        return session.query(cls).filter(cls.classes.has_key(class_name))


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


class SemSegClass(Base):
    __tablename__ = TABLENAME_SEMSEGCLASS

    id = sa.Column(sa.Integer, primary_key=True)
    tpl_id = sa.Column(sa.Integer, sa.ForeignKey(f'{TABLENAME_SEMSEGTPL}.id'))
    name = sa.Column(sa.String())
    ss_type = sa.Column(sa.Enum(MaskType))

    @classmethod
    def build_new(cls, name: str, ss_type: MaskType):
        res = cls()
        res.name = name
        res.ss_type = ss_type
        return res


class SemSegTpl(Base):
    __tablename__ = TABLENAME_SEMSEGTPL

    id = sa.Column(sa.Integer, primary_key=True)
    name = sa.Column(sa.String())
    ss_classes = relationship(SemSegClass)

    @classmethod
    def build_new(cls, tpl_name: str, seg_types: Dict[str, MaskType]):
        res = cls()
        res.name = tpl_name
        res.ss_classes = []
        for seg_type_key in seg_types:
            res.ss_classes.append(SemSegClass.build_new(seg_type_key, seg_types[seg_type_key]))
        return res


class SemSegMask(Classifiable, NumpyBinary, Base):
    __tablename__ = TABLENAME_SEM_SEG

    id = sa.Column(sa.Integer, primary_key=True)

    tpl_id = sa.Column(sa.Integer, sa.ForeignKey(f'{TABLENAME_SEMSEGTPL}.id'))
    tpl = relationship(SemSegTpl, uselist=False)

    for_frame = sa.Column(sa.Integer)
    mtype = sa.Column(sa.Enum(MaskType))
    im_stack_id = sa.Column(sa.Integer, sa.ForeignKey(f'{TABLENAME_IM_STACKS}.id'))


class ImStack(Classifiable, NumpyBinary, Base):
    __tablename__ = TABLENAME_IM_STACKS

    id = sa.Column(sa.Integer, primary_key=True)
    sbjt_id = sa.Column(sa.Integer, sa.ForeignKey(f'{TABLENAME_SUBJECTS}.id'))
    extra_info = sa.Column(pg.JSONB())

    semseg_masks: List = relationship("SemSegMask")

    @classmethod
    def build_new(cls, src_im: np.ndarray, extra_info: Dict = None):
        """
        Only adds images, not volumes or videos! Unless it is already in shape (frames, width, height, channels).
        Multi-channel images are assumed to be channels last.
        Grayscale images are assumed to be of shape (width, height).

        The array is saved using type np.uint8 and is expected to have intensities in the range of [0, 255]

        :param src_im: Numpy Array. Can be of shape (W, H), (W, H, #C) or (#F, W, H, #C)
        :param extra_info: Extra info for a human. Must contain only standard types to be json serializable
        """
        res = cls()

        if len(src_im.shape) == 2:
            # Assumption: This is a grayscale image
            src_im = np.reshape(src_im, (1, src_im.shape[0], src_im.shape[1], 1))
        elif len(src_im.shape) == 3:
            # This is the image adder function, so assume this is RGB
            src_im = np.reshape(src_im, (1, src_im.shape[0], src_im.shape[1], src_im.shape[2]))
        elif len(src_im.shape) == 4:
            # It is assumed that the array is already in correct shape
            src_im = src_im
        else:
            raise Exception("This array can not be an image, check shape!")

        assert (src_im.dtype == np.uint8 and len(src_im.shape) == 4)
        res.set_array(src_im)
        if extra_info is not None:
            res.extra_info = extra_info
        else:
            res.extra_info = {}
        return res

    def add_ss_mask(self, gt_arr: np.ndarray, sem_seg_tpl: SemSegTpl, for_frame=0, ignore_shape_mismatch=False):
        if len(gt_arr.shape) == 2:
            # This is a single indicated structure without a last dimension, add it!
            gt_arr = np.reshape(gt_arr, (gt_arr.shape[0], gt_arr.shape[1], 1))

        assert (gt_arr.dtype == np.bool), 'wrong type for a semantic segmentation mask'
        assert (len(gt_arr.shape) == 3), 'Wrong shape for a semantic segmentation mask'
        assert (ignore_shape_mismatch or self.get_ndarray().shape[1:3] == gt_arr.shape[:2]), \
            f'Shapes of seg mask {gt_arr.shape} and im stack {self.get_ndarray().shape} do not match'
        # noinspection PyTypeChecker
        tpl_classes_num = len(sem_seg_tpl.ss_classes)
        assert (gt_arr.shape[2] == tpl_classes_num), f'{gt_arr.shape[2]} classes but {tpl_classes_num} in template'
        m = SemSegMask()
        m.set_array(gt_arr)
        m.tpl = sem_seg_tpl
        m.for_frame = for_frame
        self.semseg_masks.append(m)
        return m

    def __repr__(self):
        return f'ImageStack with masks:\n{[mask for mask in self.semseg_masks]}\n{super().__repr__()}'


class Subject(Classifiable, Base):
    """
    In a medical context a subject is concerned with the data of one patient.
    For example, a patient has classes (disease_1, ...), imaging (US video, CT volumetric data, x-ray image, ...),
    text (symptom description, history) and structured data (date of birth, nationality...).

    Wherever possible the data is saved in json format, but for example for imaging only the metadata is saved
    as json, the actual image file can be found in the binaries-list.

    In future releases a complete changelog will be saved in a format suitable for process mining.
    Currently one subject can only live in one dataset, as a result a subject cannot be shared among datasets.
    """
    __tablename__ = TABLENAME_SUBJECTS

    id = sa.Column(sa.Integer(), primary_key=True)
    dataset_id = sa.Column(sa.Integer(), sa.ForeignKey(f'{TABLENAME_DATASETS}.id'))
    extra_info = sa.Column(pg.JSONB())
    name = sa.Column(sa.String())
    ims = relationship(ImStack)

    @classmethod
    def build_new(cls, name: str, extra_info: Dict = None):
        res = cls()
        res.name = name
        if extra_info is not None:
            res.extra_info = extra_info
        else:
            res.extra_info = {}
        return res


sbjts_splits_association = sa.Table(
    'sbjtsplits_association',
    Base.metadata,
    sa.Column('subject_id', sa.Integer, sa.ForeignKey(f'{TABLENAME_SUBJECTS}.id')),
    sa.Column('split_id', sa.Integer, sa.ForeignKey(f'{TABLENAME_SPLITS}.id'))
)


class Split(Base):
    __tablename__ = TABLENAME_SPLITS

    id = sa.Column(sa.Integer(), primary_key=True)
    dataset_id = sa.Column(sa.Integer(), sa.ForeignKey(f'{TABLENAME_DATASETS}.id'))
    name = sa.Column(sa.String())
    sbjts = relationship(Subject, secondary=sbjts_splits_association)

    def __len__(self):
        return len(self.sbjts)

    def __getitem__(self, item):
        return self.sbjts[item]


class Dataset(Base):
    __tablename__ = TABLENAME_DATASETS

    id = sa.Column(sa.Integer, primary_key=True)
    splits = relationship(Split)
    sbjts = relationship(Subject)
    name = sa.Column(sa.String())

    @classmethod
    def build_new(cls, name: str):
        res = cls()
        res.name = name
        return res

    def add_split(self, split_name: str):
        split = Split()
        split.name = split_name
        split.sbjts = []
        # noinspection PyUnresolvedReferences
        self.splits.append(split)

    def get_split_by_name(self, split_name: str):
        # noinspection PyTypeChecker
        for split in self.splits:
            if split.name == split_name:
                return split
        raise Exception(f"Split {split_name} does not exist")

    def get_summary(self) -> str:
        split_summary = ""
        # noinspection PyTypeChecker
        for split in self.splits:
            split_summary += f'{split}\n'
        return split_summary

    def __iter__(self):
        raise NotImplementedError()

    def __len__(self):
        # noinspection PyTypeChecker
        return len(self.sbjts)

    def __repr__(self):
        return f"{self.name} with {len(self)} subjects"


def reset_database():
    sbjts_splits_association.drop(bind=engine)
    mappers = [
        SemSegClass,
        SemSegTpl,
        SemSegMask,
        ImStack,
        Subject,
        Split,
        Dataset]
    # noinspection PyUnresolvedReferences
    Base.metadata.drop_all(bind=engine, tables=[c.__table__ for c in mappers])
    Base.metadata.create_all(engine)


Base.metadata.create_all(engine)
