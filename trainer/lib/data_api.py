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
"""

from __future__ import annotations  # Important for function annotations of symbols that are not loaded yet

import json
import os
import pickle
import shutil
import tempfile
from abc import ABC
from enum import Enum
from typing import Dict, Callable, Union, List, Tuple, Set, Any

import PySimpleGUI as sg
import numpy as np
import skimage

from trainer.lib.misc import download_and_extract


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
            BinaryType.TorchStateDict.value: BinarySaveProvider.Pickle
        }

    Unknown = 'unknown'  # pickle
    NumpyArray = 'numpy_array'
    TorchStateDict = 'torch_state_dict'


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


def build_full_filepath(name: str, dir_path: str):
    basename = os.path.splitext(name)[0]  # If a file ending is provided it is ignored
    return os.path.join(dir_path, f"{basename}.json")


def dir_is_valid_entity(dir_name: str):
    if not os.path.exists(os.path.join(dir_name, ENTITY_DIRNAME)):
        print(14)
    assert (os.path.exists(os.path.join(dir_name, ENTITY_DIRNAME)))
    return True


class Entity(ABC):
    """
    Intended to be subclassed by classes which need to persist their state
    in the form of binaries (Video, images...) and json metadata (name, attributes...).

    Every entity can have children which are also entities. For an ML folder the hierarchy looks as following:
    Dataset -> Subjects -> Data Objects (ImageStacks...)

    An entity allows inheriting classes to store files into
    TODO: Code snippet for finding out working directory.
    The only reserved names are for the json files and the folder for binaries.

    An entity writes all changes immediately to disk after the first to_disk() call.
    Beforehand, nothing is written to disk.

    Children, binaries and attributes are loaded on demand.
    """

    @staticmethod
    def get_dummy_entity(jc_name="Test Json Class"):
        """
        Intended to be used for testing functionality concerned with the basic Jsonclass.

        >>> from trainer.lib.data_api import Entity
        >>> jc = Entity.get_dummy_entity()
        >>> jc.entity_id
        'Test Json Class'
        >>> jc._get_binary('b1')  # A small array is contained in the example
        array([1, 2, 3])
        >>> jc._get_binary('obj')
        {'this': 'is', 'an': 'object'}

        :param jc_name: Name of the Jsonclass
        :return: A reference to a Jsonclass
        """
        dir_path = tempfile.gettempdir()

        res = Entity(jc_name, dir_path)
        res.to_disk(dir_path)

        res._add_attr('some_attributes', content={
            'Attribute 1': "Value 1"
        })

        res._add_bin('b1', np.array([1, 2, 3]), b_type=BinaryType.NumpyArray.value)

        res._add_bin('picture', skimage.data.retina(), b_type=BinaryType.NumpyArray.value)

        python_obj = {
            "this": "is",
            "an": "object"
        }
        res._add_bin('obj', python_obj, BinaryType.Unknown.value)

        res.to_disk(dir_path)

        return res

    def __init__(self, entity_id: str, entity_type: str, parent_folder: str = ''):
        self.entity_id, self.entity_type, self.parent_folder = entity_id, entity_type, parent_folder
        self._attrs: Dict[str, Union[Dict, None]] = {}
        self._children: Dict[str, Entity] = {}
        self._child_types: Dict[str, str] = {}
        self._binaries: Dict[str, Any] = {}
        self._binaries_model: Dict[str, Dict] = {}
        self.saved_to_disk = False

    @classmethod
    def from_disk(cls, working_dir: str) -> Entity:
        if not dir_is_valid_entity(working_dir):
            raise Exception(f"{working_dir} is not a valid directory.")

        parent_folder, entity_id = os.path.dirname(working_dir), os.path.basename(working_dir)
        with open(os.path.join(parent_folder, entity_id, ENTITY_DIRNAME, ENTITY_JSON), 'r') as f:
            json_content = json.load(f)
        res = cls(entity_id, parent_folder)
        res._attrs = {key: None for key in json_content['attrs']}
        res._child_types = json_content['children']
        res._children = {key: None for key in json_content['children']}
        res._binaries_model = json_content['bins']
        res.saved_to_disk = True
        return res

    def to_disk(self, parent_folder: str = "", properly_formatted=True) -> None:
        if parent_folder:
            self.parent_folder = parent_folder

        # The entity uses this flag to keep track of its state on disk without querying the hard drive
        self.saved_to_disk = True

        # Create the parent directory for this instance
        if not os.path.exists(self.get_working_directory()):
            os.mkdir(self.get_working_directory())

        if not os.path.exists(self._get_entity_directory()):
            os.mkdir(self._get_entity_directory())

        if not os.path.exists(self._get_bin_dir()):
            os.mkdir(self._get_bin_dir())

        self._save_json_model(properly_formatted=properly_formatted)
        self._write_binaries()
        self._save_children()

    def remove_from_disk(self):
        assert (os.path.exists(self.get_working_directory()))
        shutil.rmtree(self.get_working_directory())
        while os.path.exists(self.get_working_directory()):
            pass

    def stop_auto_save(self) -> None:
        """
        For disk intensive operations that still fit in memory it might be useful to disable the auto-save mechanism.
        Calling to_disk() will revert the operation and the entity will resume auto-saving.
        """
        self.saved_to_disk = False

    def _add_child(self, child: Entity) -> None:
        self._child_types[child.entity_id] = child.entity_type
        self._children[child.entity_id] = child
        if self._is_saved_to_disk():
            child.to_disk(self.get_working_directory())
            self._save_json_model()

    def _get_child(self, child_id: str, store_in_mem=False) -> Entity:
        if self._children[child_id] is not None:
            return self._children[child_id]

        child = Entity.from_disk(os.path.join(self.get_working_directory(), child_id))

        if self._is_saved_to_disk():
            child.parent_folder = self.get_working_directory()

        if store_in_mem:
            self._children[child_id] = child
            return self._children[child_id]
        else:
            return child

    def _save_json_model(self, properly_formatted=True):
        # Write the json model file
        save_json = {
            "attrs": list(self._attrs.keys()),
            "bins": self._binaries_model,
            "children": self._child_types
        }
        with open(self._get_json_path(), 'w+') as f:
            if properly_formatted:
                json.dump(save_json, f, indent=4)
            else:
                json.dump(save_json, f)
        for attr_id in filter(lambda x: self._attrs[x] is not None, self._attrs.keys()):
            with open(os.path.join(self.get_working_directory(), f'{attr_id}.json'), 'w+') as f:
                if properly_formatted:
                    json.dump(self._attrs[attr_id], f, indent=4)
                else:
                    json.dump(self._attrs[attr_id], f)

    def _save_children(self):
        for child_key in filter(lambda x: self._children[x] is not None, self._children.keys()):
            self._children[child_key].to_disk(parent_folder=self.get_working_directory())

    def _get_children_keys(self, entity_type='') -> List[str]:
        """
        Use for finding and iterating through children of this entity.

        :param entity_type: Filter by this entity type
        :return: The names of the children
        """
        if entity_type:
            return list(filter(lambda x: self._child_types[x] == entity_type, self._child_types.keys()))
        return list(self._child_types.keys())

    def _write_binaries(self):
        for binary_key in self._binaries:
            self._save_binary(binary_key)

    def _add_attr(self, attr_id, content=None):
        self._attrs[attr_id] = content

    def _load_attr(self, attr_id):
        """
        Returns a dict, which is a mutable python object.
        This means that changes will be saved to memory, but not implicitly to disk.
        Call to_disk() for writing changes to disk.

        :param attr_id:
        :return: The respective dictionary
        """
        if self._is_saved_to_disk() and self._attrs[attr_id] is None:
            attr_path = os.path.join(self.get_working_directory(), f'{attr_id}.json')
            if os.path.exists(attr_path):
                with open(attr_path, 'r') as f:
                    self._attrs[attr_id] = json.load(f)
            else:
                self._attrs[attr_id] = {}
        return self._attrs[attr_id]

    def _load_binary(self, binary_id) -> None:
        path_no_ext = os.path.join(self._get_bin_dir(), binary_id)
        if self._get_bin_provider(binary_id) == BinarySaveProvider.Pickle:
            with open(f'{path_no_ext}.{PICKLE_EXT}', 'rb') as f:
                binary_payload = pickle.load(f)
        else:
            binary_payload = np.load(f'{path_no_ext}.npy', allow_pickle=False)
        self._binaries[binary_id] = binary_payload

    def _get_bin_provider(self, binary_key: str):
        """
        Returns the provider for the respective binary.
        The provider is the software that is used for saving.

        >>> from trainer.lib.data_api import Entity
        >>> jc = Entity.get_dummy_entity()
        >>> jc._get_bin_provider('b1')
        <BinarySaveProvider.Numpy: 1>
        """
        return BinaryType.provider_map()[self._binaries_model[binary_key][BINARY_TYPE_KEY]]

    def _is_saved_to_disk(self) -> bool:
        if self.saved_to_disk:
            return True
        return self.parent_folder is not None and self.parent_folder and os.path.exists(self.get_working_directory())

    def _save_binary(self, bin_key) -> None:
        """
        Writes the selected binary on disk.
        :param bin_key: identifier of the binary
        """
        if not self._is_saved_to_disk():
            raise Exception(f"Before saving {bin_key}, save {self.entity_id} to disk!")
        path_no_ext = os.path.join(self._get_bin_dir(), bin_key)
        if self._get_bin_provider(bin_key) == BinarySaveProvider.Pickle:
            with open(f'{path_no_ext}.{PICKLE_EXT}', 'wb') as f:
                pickle.dump(self._binaries[bin_key], f)
        else:
            np.save(path_no_ext, self._binaries[bin_key])
        self._save_json_model()

    def _get_parent_directory(self):
        return self.parent_folder

    def get_working_directory(self):
        if self.parent_folder:
            return os.path.join(self.parent_folder, self.entity_id)
        else:
            raise Exception(f"No working directory can be stated for {self.entity_id}")

    def _get_entity_directory(self):
        return os.path.join(self.get_working_directory(), ENTITY_DIRNAME)

    def _get_bin_dir(self):
        return os.path.join(self._get_entity_directory(), BINARIES_DIRNAME)

    def _get_json_path(self):
        return os.path.join(self._get_entity_directory(), ENTITY_JSON)

    def _add_bin(self,
                 binary_id: str,
                 binary: Union[Any, np.ndarray],
                 b_type: str = BinaryType.Unknown.value,
                 meta_data=None,
                 overwrite=True) -> None:
        """
        Adds a numpy array or a pickled object.

        Note that the childclass must be saved using ```childclass.to_disk()``` to actually be written to disk.
        :param binary_id: Unique id of this binary
        :param binary: The binary content. Numpy Arrays and pickle-compatible objects are accepted.
        :param b_type:
        :param meta_data:
        :param overwrite: Overwrites existing binaries with the same id if true
        :return:
        """
        if not overwrite and binary_id in self._binaries:
            raise Exception("This binary already exists")
        self._binaries[binary_id] = binary

        self._binaries_model[binary_id] = {
            BINARY_TYPE_KEY: b_type
        }
        if meta_data is None:
            self._binaries_model[binary_id]["meta_data"] = {}
        else:
            self._binaries_model[binary_id]["meta_data"] = meta_data
        if self._is_saved_to_disk():
            self._save_binary(binary_id)
            self._save_json_model()

    def _remove_binary(self, binary_name):
        # Remove the key in model
        self._binaries_model.pop(binary_name)
        self._binaries.pop(binary_name)

        # Remove from disk if saved to disk
        if self._is_saved_to_disk():
            p = os.path.join(self._get_bin_dir(), f"{binary_name}.npy")
            os.remove(p)
            self.to_disk()

    def _get_binary(self, binary_name):
        if binary_name not in self._binaries:
            self._load_binary(binary_name)
        return self._binaries[binary_name]

    def _get_binary_model(self, binary_name):
        return self._binaries_model[binary_name]

    def _get_binary_list_filtered(self, key_filterer: Callable[[Dict], bool]):
        return [i for i in self._binaries_model if key_filterer(self._binaries_model[i])]

    def count_binaries_memory(self) -> int:
        """
        :return: The memory occupied by all the binaries together.
        """
        c_mem = 0
        for c_key in self._get_children_keys():
            c_mem += self._get_child(c_key)
        return sum([self._binaries[k].nbytes for k in self._binaries.keys()]) + c_mem

    def __str__(self):
        res = f"Entity ID: {self.entity_id}:\n"
        if self._is_saved_to_disk():
            res += f"Last saved at {self.get_working_directory()}\n"
        res += f"Binaries in memory: {list(self._binaries.keys())}\n"
        res += f'Attrs: {list(self._attrs.keys())}\n'
        res += f'Children: {list(self._children.keys())}\n'
        res += f'Bins: {list(self._binaries_model.keys())}\n'
        return res

    def __repr__(self):
        summary = str(self)
        return summary

    def __getitem__(self, item):
        if type(item) == str:
            return self._get_child(item)
        elif type(item) == int:
            return self._get_child(list(self._children.keys())[0])

    def __len__(self):
        return len(self._children)

    def __setitem__(self, key: str, value: Entity) -> None:
        """
        Adds a child entity.

        :param key: Unused parameter, existing for python consistency reason. Use anything, e.g. 0
        :param value: A child entity. Its entity_id will be used as the key.
        """
        self._add_child(value)


class ClassyEntity(Entity):
    ATTR_CLASSES = 'classes'

    def __init__(self, entity_id: str, entity_type: str, parent_folder: str = ''):
        super().__init__(entity_id, entity_type, parent_folder=parent_folder)
        self._add_attr(self.ATTR_CLASSES, content={})

    def set_class(self, class_id: str, value: str, for_dataset: Dataset = None) -> None:
        """
        Set a class to true. Classes are stored by their unique string.
        A class is only fully defined in complement with a dataset's information about that class.

        Complete absence of a class indicates an unknown.

        Hint: If two states of one class can be true to the same time, do not model them as one class.
        Instead of modelling ligament tear as one class, define a binary class for each different ligament.

        :param class_id: Unique string that is used to identify the class.
        :param value: boolean indicating
        :param for_dataset: If provided, set_class checks for compliance with the dataset.
        """
        if for_dataset is not None:
            class_obj = for_dataset.get_class(class_id)
            assert_error = f"{class_id} cannot be set to {value} according to {for_dataset.entity_id}"
            assert (value not in class_obj['values']), assert_error

        self._load_attr(self.ATTR_CLASSES)[class_id] = value

    def get_class_value(self, class_name: str):
        if class_name in self._load_attr(self.ATTR_CLASSES):
            return self._load_attr(self.ATTR_CLASSES)[class_name]
        return "--Removed--"

    def remove_class(self, class_name: str):
        self._load_attr(self.ATTR_CLASSES).pop(class_name)

    def contains_class(self, class_name: str):
        return class_name in self._load_attr(self.ATTR_CLASSES)


class ImageStack(ClassyEntity):
    SRC_KEY = 'src_im'

    def __init__(self, entity_id: str, parent_folder=''):
        super().__init__(entity_id, entity_type='image_stack', parent_folder=parent_folder)

    @classmethod
    def from_np(cls, entity_id: str, src_im: np.ndarray, extra_info: Dict = None):
        """
        Only adds images, not volumes or videos! Unless it is already in shape (frames, width, height, channels).
        Multi-channel images are assumed to be channels last.
        Grayscale images are assumed to be of shape (width, height).

        The array is saved using type np.uint8 and is expected to have intensities in the range of [0, 255]

        :param entity_id: Unique identifier of this image stack
        :param src_im: Numpy Array. Can be of shape (W, H), (W, H, #C) or (#F, W, H, #C)
        :param extra_info: Extra info for a human. Must contain only standard types to be json serializable
        """
        cls_instance = cls(entity_id)
        # Save corresponding json metadata
        meta = {}
        if len(src_im.shape) == 2:
            # Assumption: This is a grayscale image
            res = np.reshape(src_im, (1, src_im.shape[0], src_im.shape[1], 1))
            meta["image_type"] = "grayscale"
        elif len(src_im.shape) == 3:
            # This is the image adder function, so assume this is RGB
            res = np.reshape(src_im, (1, src_im.shape[0], src_im.shape[1], src_im.shape[2]))
            meta["image_type"] = "multichannel"
        elif len(src_im.shape) == 4:
            # It is assumed that the array is already in correct shape
            res = src_im
            meta["image_type"] = "video"
        else:
            raise Exception("This array can not be an image, check shape!")

        # Extra info
        if extra_info is not None:
            meta["extra"] = extra_info

        cls_instance._add_bin(cls_instance.SRC_KEY, res.astype(np.uint8), b_type=BinaryType.NumpyArray.value,
                              meta_data=meta)
        return cls_instance

    @staticmethod
    def get_sem_seg_naming_conv(sem_seg_tpl: str, frame_number=0):
        return f"gt_{sem_seg_tpl}_{frame_number}"

    def get_src(self) -> np.ndarray:
        return self._get_binary(self.SRC_KEY)

    def delete_gt(self, sem_seg_tpl: str, frame_number=0):
        print(f"Deleting ground truth of {sem_seg_tpl} at frame {frame_number}")
        self._remove_binary(self.get_sem_seg_naming_conv(sem_seg_tpl, frame_number))

    def add_sem_seg(self,
                    gt_arr: np.ndarray,
                    sem_seg_tpl: str,
                    frame_number=0) -> None:
        """
        Adds a semantic segmentation mask

        :param gt_arr: An array of type np.bool
        :param sem_seg_tpl: Key/name/identifier of the semantic segmentation template
        :param frame_number: Frame that this mask should be assigned to. Keep 0 for single images.
        """
        assert gt_arr.dtype == np.bool, "Semantic segmentation assumes binary masks!"

        if len(gt_arr.shape) == 2:
            # This is a single indicated structure without a last dimension, add it!
            gt_arr = np.reshape(gt_arr, (gt_arr.shape[0], gt_arr.shape[1], 1))

        meta = {
            "frame_number": frame_number,
            "sem_seg_tpl": sem_seg_tpl
        }
        self._add_bin(self.get_sem_seg_naming_conv(sem_seg_tpl, frame_number), gt_arr,
                      b_type=BinaryType.NumpyArray.value, meta_data=meta)

    def get_structure_list(self, image_stack_key: str = ''):
        """
        Computes the possible structures. If no image_stack_key is provided, all possible structures are returned.
        :param image_stack_key:
        :return:
        """
        if image_stack_key:
            if "structures" in self._binaries_model[image_stack_key]["meta_data"]:
                return self._binaries_model[image_stack_key]["meta_data"]["structures"]
            else:
                return []
        else:
            raise NotImplementedError()

    def get_sem_seg_frames(self, sem_seg_tpl):

        # Find out which frames contain the semantic segmentation ground truths
        frame_num = self.get_src().shape[0]
        for f_i in range(frame_num):
            pass


class Subject(ClassyEntity):
    """
    In a medical context a subject is concerned with the data of one patient.
    For example, a patient has classes (disease_1, ...), imaging (US video, CT volumetric data, x-ray image, ...),
    text (symptom description, history) and structured data (date of birth, nationality...).

    Wherever possible the data is saved in json format, but for example for imaging only the metadata is saved
    as json, the actual image file can be found in the binaries-list.

    In future releases a complete changelog will be saved in a format suitable for process mining.
    """

    def __init__(self, entity_id: str, parent_folder=''):
        super().__init__(entity_id, entity_type='subject', parent_folder=parent_folder)

    def get_image_stack_keys(self):
        return self._get_children_keys(entity_type='image_stack')
        # return self.get_binary_list_filtered(lambda x: x["binary_type"] == BinaryType.ImageStack.value)

    def add_image_stack(self, e: ImageStack):
        self._add_child(e)

    def get_image_stack(self, im_stack_key) -> ImageStack:
        res = self._get_child(im_stack_key)
        res.__class__ = ImageStack
        return res

    def get_manual_struct_segmentations(self, struct_name: str) -> Tuple[Dict[str, List[int]], int]:
        res, n = {}, 0

        def filter_imgstack_structs(x: Dict):
            is_img_stack = x['binary_type'] == BinaryType.ImageStack.value
            contains_struct = struct_name in x['meta_data']['structures']
            return is_img_stack and contains_struct

        # Iterate over image stacks that contain the structure
        for b_name in self.get_image_stack_keys():
            # Find the masks of this binary and list them
            image_stack = self._get_child(b_name)
            bs = self.get_masks_of(b_name)
            n += len(bs)
            if bs:
                res[b_name] = bs

        return res, n


class Dataset(Entity):
    ATTR_CLASSDEFINITIONS = 'class_definitions'
    ATTR_SPLITS = 'splits'
    ATTR_SEM_SEG_TPL = 'sem_seg_tpl'

    def __init__(self, name: str, parent_folder: str):
        super().__init__(name, entity_type='dataset', parent_folder=parent_folder)

    @classmethod
    def build_new(cls, name: str, dir_path: str, example_class=True):
        res = cls(name, dir_path)
        res._add_attr(res.ATTR_SPLITS, content={
            "subjects": [],
            "splits": {},
        })
        res._add_attr(res.ATTR_CLASSDEFINITIONS, content={})
        res._add_attr(res.ATTR_SEM_SEG_TPL, content={
            "basic": {"foreground": MaskType.Blob.value,
                      "outline": MaskType.Line.value}
        })
        if example_class:
            res.add_class("example_class", class_type=ClassType.Nominal,
                          values=["Unknown", "Tiger", "Elephant", "Mouse"])
        res.to_disk()
        return res

    @classmethod
    def download(cls, url: str, local_path='.', dataset_name: str = None):
        working_dir_path = download_and_extract(url, parent_dir=local_path, dir_name=dataset_name)
        return Dataset.from_disk(working_dir_path)

    def add_class(self, class_name: str, class_type: ClassType, values: List[str]):
        """
        Adds a class on a dataset level.
        This allows children to just specify a classname and from the dataset the class details can be inferred.

        :param class_name:
        :param class_type:
        :param values:
        :return:
        """
        obj = {
            "class_type": class_type.value,
            "values": values
        }
        self._load_attr(self.ATTR_CLASSDEFINITIONS)[class_name] = obj

    def get_class_names(self):
        return list(self._load_attr(self.ATTR_CLASSDEFINITIONS).keys())

    def get_class(self, class_name: str) -> Union[Dict, None]:
        if class_name in self._load_attr(self.ATTR_CLASSDEFINITIONS):
            return self._load_attr(self.ATTR_CLASSDEFINITIONS)[class_name]
        else:
            return None

    def remove_class(self, class_name: str):
        self._load_attr(self.ATTR_CLASSDEFINITIONS).pop(class_name)

    def get_structure_template_names(self):
        return list(self._load_attr(self.ATTR_CLASSDEFINITIONS).keys())

    def get_structure_template_by_name(self, tpl_name):
        return self._load_attr(self.ATTR_SEM_SEG_TPL)[tpl_name]

    def save_subject(self, s: Subject) -> None:
        """
        Creates a new subject in this dataset

        :param s: Unique identifier of the new subject
        """
        self._add_child(s)
        # Add the name of the subject into the splits
        if s.entity_id not in self._load_attr(self.ATTR_SPLITS)['subjects']:
            self._load_attr(self.ATTR_SPLITS)["subjects"].append(s.entity_id)

    def get_subject_name_list(self, split='') -> List[str]:
        """
        Computes the list of subjects in this dataset.
        :param split: Dataset splits of the subjects
        :return: List of the names of the subjects
        """
        if not split:
            subjects = self._get_children_keys(entity_type='subject')
        else:
            subjects = self._load_attr(self.ATTR_SPLITS)["splits"][split]
        return subjects

    def append_subject_to_split(self, s_id: str, split: str):
        # Create the split if it does not exist
        if split not in self._load_attr(self.ATTR_SPLITS)["splits"]:
            self._load_attr(self.ATTR_SPLITS)["splits"][split] = []

        self._load_attr(self.ATTR_SPLITS)["splits"][split].append(s_id)

    def filter_subjects(self, filterer: Callable[[Subject], bool], viz=False) -> List[str]:
        """
        Returns a list with the names of subjects of interest.
        :param filterer: If the filterer returns true, the subject is added to the list
        :param viz: Whether or not a progress meter should be displayed
        :return: The list of subjects of interest
        """
        res: List[str] = []
        for i, s_name in enumerate(self._load_attr(self.ATTR_SPLITS)["subjects"]):
            te = self.get_subject_by_name(s_name)
            if filterer(te):
                res.append(te.entity_id)
            if viz:
                sg.OneLineProgressMeter("Filtering subjects", i + 1,
                                        self.get_subject_count(),
                                        'key',
                                        f'Subject: {te.entity_id}')
        return res

    def get_subject_by_name(self, s_name: str) -> Subject:
        if s_name not in self._load_attr(self.ATTR_SPLITS)['subjects']:
            raise Exception('This dataset does not contain a subject with this name')
        res = self._get_child(s_name)
        res.__class__ = Subject
        return res

    def get_summary(self) -> str:
        split_summary = ""
        for split in self._load_attr(self.ATTR_SPLITS)["splits"]:
            split_summary += f"""{split}: {self.get_subject_count(split=split)}\n"""
        return f"Saved at {self.get_working_directory()}\nN: {len(self)}\n{split_summary}"

    def compute_segmentation_structures(self) -> Dict[str, Set[str]]:
        """
        Returns a dictionary.
        Keys: All different structures.
        Values: The names of the subjects that can be used to train these structures with.
        :return: Dictionary of structures and corresponding subjects
        """
        # Segmentation Helper
        seg_structs: Dict[str, Set[str]] = {}  # structure_name: List_of_Training_Example_names with that structure

        def te_filterer(te: Subject) -> bool:
            """
            Can be used to hijack the functional filtering utility
            and uses a side effect of struct_appender to fill seg_structs.
            """

            def struct_appender(b: Dict) -> bool:
                if b['binary_type'] == BinaryType.ImageStack.value:
                    structures = list(b['meta_data']['structures'].keys())
                    for structure in structures:
                        if structure not in seg_structs:
                            seg_structs[structure] = set()
                        if te.entity_id not in seg_structs[structure]:
                            seg_structs[structure] = seg_structs[structure] | {te.entity_id}
                return True

            stacks = te._get_binary_list_filtered(struct_appender)
            return len(stacks) != 0

        self.filter_subjects(lambda x: te_filterer(x))
        return seg_structs

    def get_subject_count(self, split=''):
        return len(self.get_subject_name_list(split=split))

    def save_model_state(self, weight_id: str, binary: Any) -> None:
        self._add_bin(
            weight_id,
            binary,
            BinaryType.TorchStateDict.value
        )

    def __len__(self):
        return self.get_subject_count()

    def __getitem__(self, item):
        return self.get_subject_by_name(item)

    def __iter__(self):
        """
        Iterates through the subjects of this dataset
        """

        s_ls = self.get_subject_name_list()
        for s_key in s_ls:
            yield self.get_subject_by_name(s_key)
