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

from functools import reduce
import json
import os
import pathlib
from ast import literal_eval as make_tuple
from enum import Enum
from typing import List, Dict, Union
import random

import numpy as np
import sqlalchemy as sa
from sqlalchemy import event
import sqlalchemy.dialects.postgresql as pg
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

import trainer.lib as lib

engine = create_engine(lib.config[lib.DB_CON_KEY])
Session = sessionmaker(bind=engine)

Base = declarative_base()

TABLENAME_CLASSDEFINITIONS = 'classdefinitions'
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
    file_path = sa.Column(sa.String())
    stored_in_db = sa.Column(sa.Boolean())

    def __init__(self):
        self.tmp_arr: Union[np.ndarray, None] = None

    @sa.orm.reconstructor
    def init_on_load(self):
        """
        Does the job of the constructor in case of an object which is loaded from the database.
        See https://docs.sqlalchemy.org/en/13/orm/constructors.html for details.
        """
        self.tmp_arr: Union[np.ndarray, None] = None

    def get_bin_disk_path(self):
        if not self.file_path:
            existing = os.listdir(lib.config[lib.BIG_BIN_KEY])
            shape_str = reduce(lambda x, y: f'{x}_{y}', make_tuple(f'({self.shape})'))
            proposal: str = f'NPY_{self.dtype}_{shape_str}'
            while f'{proposal}.npy' in existing:
                proposal += f'_{random.randint(0, 50000):05d}'
            self.file_path = os.path.join(lib.config[lib.BIG_BIN_KEY], f'{proposal}.npy')
        return self.file_path

    def set_array(self, arr: np.ndarray) -> None:
        self.tmp_arr = arr
        self.shape = str(arr.shape)[1:-1]
        self.dtype = str(arr.dtype)
        if arr.size * 8 < 1024 * 1024 * 1024:
            self.binary = arr.tobytes()
            self.stored_in_db = True
        else:
            # Array is too large to be stored using Postgresql bytea column
            np.save(self.get_bin_disk_path(), arr)
            self.stored_in_db = False

    def get_ndarray(self) -> np.ndarray:
        if self.tmp_arr is None:
            if self.stored_in_db:
                self.tmp_arr = np.frombuffer(self.binary, dtype=self.dtype).reshape(
                    make_tuple(f'({self.shape})')).copy()
            else:
                self.tmp_arr = np.load(self.file_path)
        return self.tmp_arr

    def __repr__(self):
        return f"Numpy Binary with shape ({self.shape}) and type {self.dtype}>"


class ClassType(Enum):
    Binary = 'binary'
    Nominal = 'nominal'
    Ordinal = 'ordinal'


class ClassDefinition(Base):
    __tablename__ = TABLENAME_CLASSDEFINITIONS

    id = sa.Column(sa.Integer, primary_key=True)
    name = sa.Column(sa.String())
    cls_type = sa.Column(sa.Enum(ClassType))
    values = sa.Column(pg.JSONB())

    @classmethod
    def build_new(cls, name: str, cls_type: ClassType, values: List[str]):
        res = cls()
        res.name = name
        res.cls_type = cls_type
        res.values = values
        return res

    def __repr__(self):
        return f"Class definition of {self.name} with type: {self.cls_type}\nValues: {self.values}"


class Classifiable:
    classes = sa.Column(pg.JSONB())

    # definition_id = sa.Column(sa.Integer, sa.ForeignKey(f'{TABLENAME_CLASSDEFINITIONS}.id'))
    # definition = relationship(ClassDefinition)

    def set_class(self, class_name: str, class_val: str):
        if self.classes:
            self.classes[class_name] = class_val
        else:
            self.classes = {class_name: class_val}

    def remove_class(self, class_name: str):
        self.classes.pop(class_name)

    def get_class(self, class_name: str):  # -> Union[Dict, None]:
        if self.classes and class_name in self.classes:
            res = self.classes[class_name]
            return res
        else:
            return None

    @classmethod
    def query_all_with_class(cls, session: sa.orm.session.Session, class_name: str):
        return session.query(cls).filter(class_name in cls.classes)


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

    def __repr__(self):
        return f'SemSegClass {self.name} of type {self.ss_type}'


class SemSegTpl(Base):
    __tablename__ = TABLENAME_SEMSEGTPL

    id = sa.Column(sa.Integer, primary_key=True)
    name = sa.Column(sa.String())
    ss_classes: List[SemSegClass] = relationship(SemSegClass)

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

    def __repr__(self):
        return f"Mask for frame {self.for_frame} for template {self.tpl.name}"


class ImStack(Classifiable, NumpyBinary, Base):
    __tablename__ = TABLENAME_IM_STACKS

    id = sa.Column(sa.Integer, primary_key=True)
    sbjt_id = sa.Column(sa.Integer, sa.ForeignKey(f'{TABLENAME_SUBJECTS}.id'))
    extra_info = sa.Column(pg.JSONB())

    semseg_masks: List[SemSegMask] = relationship("SemSegMask")

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

    The extra_info attribute can be used freely for a json dict.

    In future releases a complete changelog will be saved in a format suitable for process mining.
    """
    __tablename__ = TABLENAME_SUBJECTS

    id = sa.Column(sa.Integer(), primary_key=True)
    extra_info = sa.Column(pg.JSONB())
    name = sa.Column(sa.String())
    ims: List[ImStack] = relationship(ImStack)

    @classmethod
    def build_new(cls, name: str, extra_info: Dict = None):
        res = cls()
        res.name = name
        if extra_info is not None:
            res.extra_info = extra_info
        else:
            res.extra_info = {}
        return res

    def __repr__(self):
        return f'Subject {self.name} with {len(self.ims)} image stacks'


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
    sbjts: List[Subject] = relationship(Subject, secondary=sbjts_splits_association)

    def __len__(self):
        return len(self.sbjts)

    def __getitem__(self, item):
        return self.sbjts[item]

    def __repr__(self):
        return f'Split {self.name} with {len(self.sbjts)} subjects'


class Dataset(Base):
    """
    A dataset is a collection of splits.
    """
    __tablename__ = TABLENAME_DATASETS

    id = sa.Column(sa.Integer, primary_key=True)
    splits: List[Split] = relationship(Split)
    name = sa.Column(sa.String())

    @classmethod
    def build_new(cls, name: str):
        res = cls()
        res.name = name
        return res

    def add_split(self, split_name: str) -> Split:
        split = Split()
        split.name = split_name
        split.sbjts = []
        self.splits.append(split)
        return split

    def get_split_by_name(self, split_name: str):
        for split in self.splits:
            if split.name == split_name:
                return split
        raise Exception(f"Split {split_name} does not exist")

    def get_summary(self) -> str:
        split_summary = ""
        for split in self.splits:
            split_summary += f'{split}\n'
        return split_summary

    def __iter__(self):
        raise NotImplementedError()

    def __len__(self):
        # TODO: Handle the case of the same subject being in different splits
        return sum([len(split.sbjts) for split in self.splits])

    def __repr__(self):
        return f"{self.name} with {len(self)} subjects\nSplits: {self.splits}"


def reset_database():
    # Reset storage on disk
    from trainer.lib.misc import delete_dir
    bin_dir_path = lib.config[lib.BIG_BIN_KEY]
    print(f"Deleting {len(os.listdir(bin_dir_path))} binaries from {bin_dir_path}")
    delete_dir(bin_dir_path)

    sbjts_splits_association.drop(bind=engine)
    mappers = [
        ClassDefinition,
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


# event.listen(Session, "before_commit", NumpyBinary.commit_handler)

Base.metadata.create_all(engine)
