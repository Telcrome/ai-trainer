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
from typing import Any

import numpy as np
import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# engine = create_engine('sqlite:///:memory:', echo=True)
engine = create_engine('sqlite:///./test.db')
Session = sessionmaker(bind=engine)

Base = declarative_base()


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


class NumpyBinary(Base):
    __tablename__ = 'npy_binaries'

    id = sa.Column(sa.Integer, primary_key=True)
    binary = sa.Column(sa.LargeBinary)
    shape = sa.Column(sa.String())
    dtype = sa.Column(sa.String())

    def __init__(self, binary: Any, shape: str, dtype: str):
        self.binary, self.shape, self.dtype = binary, shape, dtype

    @classmethod
    def from_ndarray(cls, arr: np.ndarray):
        return cls(binary=arr.tobytes(), shape=str(arr.shape)[1:-1], dtype=str(arr.dtype))

    def get_ndarray(self) -> np.ndarray:
        return np.frombuffer(self.binary, dtype=self.dtype).reshape(make_tuple(f'({self.shape})'))

    def __repr__(self):
        return f"<Numpy array with shape ({self.shape}) and type {self.dtype}>"


# def __init__(self, arr: np.ndarray):
#     self.shape = str(arr.shape)
#     self.type = NumpyType
#     self.binary = arr.tobytes()


class ClassType(Enum):
    Binary = 'binary'
    Nominal = 'nominal'
    Ordinal = 'ordinal'


class ClassSelectionLevel(Enum):
    SubjectLevel = "Subject Level"
    BinaryLevel = "Binary Level"
    FrameLevel = "Frame Level"


ENTITY_DIRNAME = '.entity'
BINARIES_DIRNAME = "binaries"
ENTITY_JSON = "entity.json"
BINARY_TYPE_KEY = "binary_type"
PICKLE_EXT = 'pickle'

# class Classifiable(ABC):
#     ATTR_CLASSES = 'classes'
#
#     def __init__(self, entity_id: str, entity_type: str, parent_folder: str = ''):
#         super().__init__(entity_id, entity_type, parent_folder=parent_folder)
#         self._add_attr(self.ATTR_CLASSES, content={})
#
#     def set_class(self, class_id: str, value: str, for_dataset: Dataset = None) -> None:
#         """
#         Set a class to true. Classes are stored by their unique string.
#         A class is only fully defined in complement with a dataset's information about that class.
#
#         Complete absence of a class indicates an unknown.
#
#         Hint: If two states of one class can be true to the same time, do not model them as one class.
#         Instead of modelling ligament tear as one class, define a binary class for each different ligament.
#
#         :param class_id: Unique string that is used to identify the class.
#         :param value: boolean indicating
#         :param for_dataset: If provided, set_class checks for compliance with the dataset.
#         """
#         if for_dataset is not None:
#             class_obj = for_dataset.get_class(class_id)
#             assert_error = f"{class_id} cannot be set to {value} according to {for_dataset.entity_id}"
#             assert (value not in class_obj['values']), assert_error
#
#         self._load_attr(self.ATTR_CLASSES)[class_id] = value
#
#     def get_class_value(self, class_name: str):
#         if class_name in self._load_attr(self.ATTR_CLASSES):
#             return self._load_attr(self.ATTR_CLASSES)[class_name]
#         return "--Removed--"
#
#     def remove_class(self, class_name: str):
#         self._load_attr(self.ATTR_CLASSES).pop(class_name)
#
#     def contains_class(self, class_name: str):
#         return class_name in self._load_attr(self.ATTR_CLASSES)
#
#
# class ImageStack(Base):
#     __tablename__ = 'imagestacks'
#
#     @classmethod
#     def from_np(cls, entity_id: str, src_im: np.ndarray, extra_info: Dict = None):
#         """
#         Only adds images, not volumes or videos! Unless it is already in shape (frames, width, height, channels).
#         Multi-channel images are assumed to be channels last.
#         Grayscale images are assumed to be of shape (width, height).
#
#         The array is saved using type np.uint8 and is expected to have intensities in the range of [0, 255]
#
#         :param entity_id: Unique identifier of this image stack
#         :param src_im: Numpy Array. Can be of shape (W, H), (W, H, #C) or (#F, W, H, #C)
#         :param extra_info: Extra info for a human. Must contain only standard types to be json serializable
#         """
#         cls_instance = cls(entity_id)
#         # Save corresponding json metadata
#         meta = {}
#         if len(src_im.shape) == 2:
#             # Assumption: This is a grayscale image
#             res = np.reshape(src_im, (1, src_im.shape[0], src_im.shape[1], 1))
#             meta["image_type"] = "grayscale"
#         elif len(src_im.shape) == 3:
#             # This is the image adder function, so assume this is RGB
#             res = np.reshape(src_im, (1, src_im.shape[0], src_im.shape[1], src_im.shape[2]))
#             meta["image_type"] = "multichannel"
#         elif len(src_im.shape) == 4:
#             # It is assumed that the array is already in correct shape
#             res = src_im
#             meta["image_type"] = "video"
#         else:
#             raise Exception("This array can not be an image, check shape!")
#
#         # Extra info
#         if extra_info is not None:
#             meta["extra"] = extra_info
#
#         cls_instance._add_bin(cls_instance.SRC_KEY, res.astype(np.uint8), b_type=BinaryType.NumpyArray.value,
#                               meta_data=meta)
#         return cls_instance
#
#     @staticmethod
#     def get_sem_seg_naming_conv(sem_seg_tpl: str, frame_number=0):
#         return f"gt_{sem_seg_tpl}_{frame_number}"
#
#     def get_src(self) -> np.ndarray:
#         return self._get_binary(self.SRC_KEY)
#
#     def delete_gt(self, sem_seg_tpl: str, frame_number=0):
#         print(f"Deleting ground truth of {sem_seg_tpl} at frame {frame_number}")
#         self._remove_binary(self.get_sem_seg_naming_conv(sem_seg_tpl, frame_number))
#
#     def add_sem_seg(self,
#                     gt_arr: np.ndarray,
#                     sem_seg_tpl: str,
#                     frame_number=0) -> None:
#         """
#         Adds a semantic segmentation mask
#
#         :param gt_arr: An array of type np.bool
#         :param sem_seg_tpl: Key/name/identifier of the semantic segmentation template
#         :param frame_number: Frame that this mask should be assigned to. Keep 0 for single images.
#         """
#         assert gt_arr.dtype == np.bool, "Semantic segmentation assumes binary masks!"
#
#         if len(gt_arr.shape) == 2:
#             # This is a single indicated structure without a last dimension, add it!
#             gt_arr = np.reshape(gt_arr, (gt_arr.shape[0], gt_arr.shape[1], 1))
#
#         meta = {
#             "frame_number": frame_number,
#             "sem_seg_tpl": sem_seg_tpl
#         }
#         self._add_bin(self.get_sem_seg_naming_conv(sem_seg_tpl, frame_number), gt_arr,
#                       b_type=BinaryType.NumpyArray.value, meta_data=meta)
#
#     def get_structure_list(self, image_stack_key: str = ''):
#         """
#         Computes the possible structures. If no image_stack_key is provided, all possible structures are returned.
#         :param image_stack_key:
#         :return:
#         """
#         if image_stack_key:
#             if "structures" in self._binaries_model[image_stack_key]["meta_data"]:
#                 return self._binaries_model[image_stack_key]["meta_data"]["structures"]
#             else:
#                 return []
#         else:
#             raise NotImplementedError()
#
#     def get_sem_seg_frames(self, sem_seg_tpl):
#
#         # Find out which frames contain the semantic segmentation ground truths
#         frame_num = self.get_src().shape[0]
#         for f_i in range(frame_num):
#             pass
#
#
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


Base.metadata.create_all(engine)
