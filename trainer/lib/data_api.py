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
con_string = 'postgresql+psycopg2://postgres:!supi1324!@127.0.0.1:5432/test3'
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
        if extra_info:
            res.extra_info = extra_info
        return res

    def add_ss_mask(self, gt_arr: np.ndarray, sem_seg_tpl: SemSegTpl, for_frame=0):
        if len(gt_arr.shape) == 2:
            # This is a single indicated structure without a last dimension, add it!
            gt_arr = np.reshape(gt_arr, (gt_arr.shape[0], gt_arr.shape[1], 1))

        assert (gt_arr.dtype == np.bool), 'wrong type for a semantic segmentation mask'
        assert (len(gt_arr.shape) == 3), 'Wrong shape for a semantic segmentation mask'
        assert (self.get_ndarray().shape[1:3] == gt_arr.shape[:2]), \
            f'Shapes of seg mask {gt_arr.shape} and im stack {self.get_ndarray().shape} do not match'
        # noinspection PyTypeChecker
        tpl_classes_num = len(sem_seg_tpl.ss_classes)
        assert (gt_arr.shape[2] == tpl_classes_num), f'{gt_arr.shape[2]} classes but {tpl_classes_num} in template'
        m = SemSegMask()
        m.set_array(gt_arr)
        m.tpl = sem_seg_tpl
        m.for_frame = for_frame
        return m

    def __repr__(self):
        return f'ImageStack with masks:\n{[mask for mask in self.semseg_masks]}\n{super().__repr__()}'

# class Subject(ClassyEntity):
#     """
#     In a medical context a subject is concerned with the data of one patient.
#     For example, a patient has classes (disease_1, ...), imaging (US video, CT volumetric data, x-ray image, ...),
#     text (symptom description, history) and structured data (date of birth, nationality...).
#
#     Wherever possible the data is saved in json format, but for example for imaging only the metadata is saved
#     as json, the actual image file can be found in the binaries-list.
#
#     In future releases a complete changelog will be saved in a format suitable for process mining.
#     """
#
#     def __init__(self, entity_id: str, parent_folder=''):
#         super().__init__(entity_id, entity_type='subject', parent_folder=parent_folder)
#
#     def get_image_stack_keys(self):
#         return self._get_children_keys(entity_type='image_stack')
#         # return self.get_binary_list_filtered(lambda x: x["binary_type"] == BinaryType.ImageStack.value)
#
#     def add_image_stack(self, e: ImageStack):
#         self._add_child(e)
#
#     def get_image_stack(self, im_stack_key) -> ImageStack:
#         res = self._get_child(im_stack_key)
#         res.__class__ = ImageStack
#         return res
#
#     def get_manual_struct_segmentations(self, struct_name: str) -> Tuple[Dict[str, List[int]], int]:
#         res, n = {}, 0
#
#         def filter_imgstack_structs(x: Dict):
#             is_img_stack = x['binary_type'] == BinaryType.ImageStack.value
#             contains_struct = struct_name in x['meta_data']['structures']
#             return is_img_stack and contains_struct
#
#         # Iterate over image stacks that contain the structure
#         for b_name in self.get_image_stack_keys():
#             # Find the masks of this binary and list them
#             image_stack = self._get_child(b_name)
#             bs = self.get_masks_of(b_name)
#             n += len(bs)
#             if bs:
#                 res[b_name] = bs
#
#         return res, n
#
#
# class Dataset(Entity):
#     ATTR_CLASSDEFINITIONS = 'class_definitions'
#     ATTR_SPLITS = 'splits'
#     ATTR_SEM_SEG_TPL = 'sem_seg_tpl'
#
#     def __init__(self, name: str, parent_folder: str):
#         super().__init__(name, entity_type='dataset', parent_folder=parent_folder)
#
#     @classmethod
#     def build_new(cls, name: str, dir_path: str, example_class=True):
#         res = cls(name, dir_path)
#         res._add_attr(res.ATTR_SPLITS, content={
#             "subjects": [],
#             "splits": {},
#         })
#         res._add_attr(res.ATTR_CLASSDEFINITIONS, content={})
#         res._add_attr(res.ATTR_SEM_SEG_TPL, content={
#             "basic": {"foreground": MaskType.Blob.value,
#                       "outline": MaskType.Line.value}
#         })
#         if example_class:
#             res.add_class("example_class", class_type=ClassType.Nominal,
#                           values=["Unknown", "Tiger", "Elephant", "Mouse"])
#         res.to_disk()
#         return res
#
#     @classmethod
#     def download(cls, url: str, local_path='.', dataset_name: str = None):
#         working_dir_path = download_and_extract(url, parent_dir=local_path, dir_name=dataset_name)
#         return Dataset.from_disk(working_dir_path)
#
#     def add_class(self, class_name: str, class_type: ClassType, values: List[str]):
#         """
#         Adds a class on a dataset level.
#         This allows children to just specify a classname and from the dataset the class details can be inferred.
#
#         :param class_name:
#         :param class_type:
#         :param values:
#         :return:
#         """
#         obj = {
#             "class_type": class_type.value,
#             "values": values
#         }
#         self._load_attr(self.ATTR_CLASSDEFINITIONS)[class_name] = obj
#
#     def get_class_names(self):
#         return list(self._load_attr(self.ATTR_CLASSDEFINITIONS).keys())
#
#     def get_class(self, class_name: str) -> Union[Dict, None]:
#         if class_name in self._load_attr(self.ATTR_CLASSDEFINITIONS):
#             return self._load_attr(self.ATTR_CLASSDEFINITIONS)[class_name]
#         else:
#             return None
#
#     def remove_class(self, class_name: str):
#         self._load_attr(self.ATTR_CLASSDEFINITIONS).pop(class_name)
#
#     def get_structure_template_names(self):
#         return list(self._load_attr(self.ATTR_CLASSDEFINITIONS).keys())
#
#     def get_structure_template_by_name(self, tpl_name):
#         return self._load_attr(self.ATTR_SEM_SEG_TPL)[tpl_name]
#
#     def save_subject(self, s: Subject) -> None:
#         """
#         Creates a new subject in this dataset
#
#         :param s: Unique identifier of the new subject
#         """
#         self._add_child(s)
#         # Add the name of the subject into the splits
#         if s.entity_id not in self._load_attr(self.ATTR_SPLITS)['subjects']:
#             self._load_attr(self.ATTR_SPLITS)["subjects"].append(s.entity_id)
#
#     def get_subject_name_list(self, split='') -> List[str]:
#         """
#         Computes the list of subjects in this dataset.
#         :param split: Dataset splits of the subjects
#         :return: List of the names of the subjects
#         """
#         if not split:
#             subjects = self._get_children_keys(entity_type='subject')
#         else:
#             subjects = self._load_attr(self.ATTR_SPLITS)["splits"][split]
#         return subjects
#
#     def append_subject_to_split(self, s_id: str, split: str):
#         # Create the split if it does not exist
#         if split not in self._load_attr(self.ATTR_SPLITS)["splits"]:
#             self._load_attr(self.ATTR_SPLITS)["splits"][split] = []
#
#         self._load_attr(self.ATTR_SPLITS)["splits"][split].append(s_id)
#
#     def filter_subjects(self, filterer: Callable[[Subject], bool], viz=False) -> List[str]:
#         """
#         Returns a list with the names of subjects of interest.
#         :param filterer: If the filterer returns true, the subject is added to the list
#         :param viz: Whether or not a progress meter should be displayed
#         :return: The list of subjects of interest
#         """
#         res: List[str] = []
#         for i, s_name in enumerate(self._load_attr(self.ATTR_SPLITS)["subjects"]):
#             te = self.get_subject_by_name(s_name)
#             if filterer(te):
#                 res.append(te.entity_id)
#             if viz:
#                 sg.OneLineProgressMeter("Filtering subjects", i + 1,
#                                         self.get_subject_count(),
#                                         'key',
#                                         f'Subject: {te.entity_id}')
#         return res
#
#     def get_subject_by_name(self, s_name: str) -> Subject:
#         if s_name not in self._load_attr(self.ATTR_SPLITS)['subjects']:
#             raise Exception('This dataset does not contain a subject with this name')
#         res = self._get_child(s_name)
#         res.__class__ = Subject
#         return res
#
#     def get_summary(self) -> str:
#         split_summary = ""
#         for split in self._load_attr(self.ATTR_SPLITS)["splits"]:
#             split_summary += f"""{split}: {self.get_subject_count(split=split)}\n"""
#         return f"Saved at {self.get_working_directory()}\nN: {len(self)}\n{split_summary}"
#
#     def compute_segmentation_structures(self) -> Dict[str, Set[str]]:
#         """
#         Returns a dictionary.
#         Keys: All different structures.
#         Values: The names of the subjects that can be used to train these structures with.
#         :return: Dictionary of structures and corresponding subjects
#         """
#         # Segmentation Helper
#         seg_structs: Dict[str, Set[str]] = {}  # structure_name: List_of_Training_Example_names with that structure
#
#         def te_filterer(te: Subject) -> bool:
#             """
#             Can be used to hijack the functional filtering utility
#             and uses a side effect of struct_appender to fill seg_structs.
#             """
#
#             def struct_appender(b: Dict) -> bool:
#                 if b['binary_type'] == BinaryType.ImageStack.value:
#                     structures = list(b['meta_data']['structures'].keys())
#                     for structure in structures:
#                         if structure not in seg_structs:
#                             seg_structs[structure] = set()
#                         if te.entity_id not in seg_structs[structure]:
#                             seg_structs[structure] = seg_structs[structure] | {te.entity_id}
#                 return True
#
#             stacks = te._get_binary_list_filtered(struct_appender)
#             return len(stacks) != 0
#
#         self.filter_subjects(lambda x: te_filterer(x))
#         return seg_structs
#
#     def get_subject_count(self, split=''):
#         return len(self.get_subject_name_list(split=split))
#
#     def save_model_state(self, weight_id: str, binary: Any) -> None:
#         self._add_bin(
#             weight_id,
#             binary,
#             BinaryType.Pickle.value
#         )
#
#     def __len__(self):
#         return self.get_subject_count()
#
#     def __getitem__(self, item):
#         return self.get_subject_by_name(item)
#
#     def __iter__(self):
#         """
#         Iterates through the subjects of this dataset
#         """
#
#         s_ls = self.get_subject_name_list()
#         for s_key in s_ls:
#             yield self.get_subject_by_name(s_key)

def reset_schema():
    mappers = [
        SemSegClass,
        SemSegTpl,
        SemSegMask,
        ImStack]
    # noinspection PyUnresolvedReferences
    Base.metadata.drop_all(bind=engine, tables=[c.__table__ for c in mappers])
    Base.metadata.create_all(engine)


Base.metadata.create_all(engine)
