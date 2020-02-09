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
from enum import Enum
from typing import Dict, Callable, Union, List, Tuple, Set, Any

import PySimpleGUI as sg
import numpy as np
from tqdm import tqdm

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
    assert (os.path.exists(os.path.join(dir_name, ENTITY_DIRNAME)))
    return True


class Entity:
    """
    Intended to be subclassed by classes which need to persist their state
    in the form of binaries (Video, images...) and json metadata (name, attributes...).

    Every entity can have children which are also entities. For an ML folder the hierarchy looks as following:
    Dataset -> Subjects -> Data Objects (ImageStacks...)

    An entity allows inheriting classes to store files into
    TODO: Code snippet for finding out working directory.
    The only reserved names are for the json files and the folder for binaries.
    """

    def __init__(self, entity_id: str, parent_folder: str):

        self.entity_id = entity_id
        self._attrs: Dict[str, Union[Dict, None]] = {}
        self._children: Dict[str, Entity] = {}
        self._binaries: Dict[str, Any] = {}
        self._binaries_model: Dict[str, Dict] = {}
        self.parent_folder = parent_folder
        if not os.path.exists(self.get_working_directory()):
            self.to_disk()  # Creates directories

    @classmethod
    def from_disk(cls, working_dir: str) -> Entity:
        if not dir_is_valid_entity(working_dir):
            raise Exception(f"{working_dir} is not a valid directory.")

        parent_folder, entity_id = os.path.dirname(working_dir), os.path.basename(working_dir)
        with open(os.path.join(parent_folder, entity_id, ENTITY_DIRNAME, ENTITY_JSON), 'r') as f:
            json_content = json.load(f)
        res = cls(entity_id, parent_folder)
        res._attrs = {key: None for key in json_content['attrs']}
        res._children = {key: None for key in json_content['children']}
        res._binaries_model = json_content['bins']
        # json_file_paths = os.path.join(working_dir, JSON_MODEL_FILENAME)

        return res

    def to_disk(self, parent_folder: str = "", properly_formatted=True) -> None:
        if parent_folder:
            self.parent_folder = parent_folder

        # Create the parent directory for this instance
        if not os.path.exists(self.get_working_directory()):
            os.mkdir(self.get_working_directory())

        if not os.path.exists(self.get_entity_directory()):
            os.mkdir(self.get_entity_directory())

        self._save_json_model(properly_formatted=properly_formatted)
        self._write_binaries()
        self._save_children()

    def create_child(self, entity_id: str):
        # self.to_disk()
        child = Entity(entity_id, self.get_working_directory())
        child.parent = self
        self._children[child.entity_id] = child
        return child

    def get_child(self, child_id: str, store_in_mem=False) -> Entity:
        if self._children[child_id] is not None:
            return self._children[child_id]

        child = Entity.from_disk(os.path.join(self.get_working_directory(), child_id))
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
            "children": list(self._children.keys())
        }
        with open(self.get_json_path(), 'w+') as f:
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

    def _write_binaries(self):
        if not os.path.exists(self.get_bin_dir()):
            os.mkdir(self.get_bin_dir())
        for binary_key in self._binaries:  # TODO: check if necessary
            self.save_binary(binary_key)

    def add_attr(self, attr_id, content=None):
        self._attrs[attr_id] = content
        self._save_json_model()

    def load_attr(self, attr_id):
        """
        Returns a dict, which is a mutable python object.
        This means that changes will be saved to memory, but not implicitly to disk.
        Call to_disk() for writing changes to disk.

        :param attr_id:
        :return: The respective dictionary
        """
        if self._attrs[attr_id] is None:
            attr_path = os.path.join(self.get_working_directory(), f'{attr_id}.json')
            if os.path.exists(attr_path):
                with open(attr_path, 'r') as f:
                    self._attrs[attr_id] = json.load(f)
            else:
                self._attrs[attr_id] = {}
        return self._attrs[attr_id]

    def load_binary(self, binary_id) -> None:
        path_no_ext = os.path.join(self.get_bin_dir(), binary_id)
        if self.get_bin_provider(binary_id) == BinarySaveProvider.Pickle:
            with open(f'{path_no_ext}.{PICKLE_EXT}', 'rb') as f:
                binary_payload = pickle.load(f)
        else:
            binary_payload = np.load(f'{path_no_ext}.npy', allow_pickle=False)
        self._binaries[binary_id] = binary_payload

    def get_bin_provider(self, binary_key: str):
        """
        Returns the provider for the respective binary.
        The provider is the software that is used for saving.

        >>> import trainer.lib as lib
        >>> jc = lib.get_dummy_jsonclass()
        >>> jc.get_bin_provider('b1')
        <BinarySaveProvider.Numpy: 1>
        """
        return BinaryType.provider_map()[self._binaries_model[binary_key][BINARY_TYPE_KEY]]

    def save_binary(self, bin_key) -> None:
        """
        Writes the selected binary on disk.
        :param bin_key: identifier of the binary
        """
        if self.parent_folder is None:
            raise Exception(f"Before saving {bin_key}, save {self.entity_id} to disk!")
        path_no_ext = os.path.join(self.get_bin_dir(), bin_key)
        if self.get_bin_provider(bin_key) == BinarySaveProvider.Pickle:
            with open(f'{path_no_ext}.{PICKLE_EXT}', 'wb') as f:
                pickle.dump(self._binaries[bin_key], f)
        else:
            np.save(path_no_ext, self._binaries[bin_key])
        self._save_json_model()

    def delete_on_disk(self, blocking=True):
        shutil.rmtree(self.get_working_directory(), ignore_errors=True)
        if blocking:
            while os.path.exists(self.get_working_directory()):
                pass

    def get_parent_directory(self):
        return self.parent_folder

    def get_working_directory(self):
        return os.path.join(self.parent_folder, self.entity_id)

    def get_entity_directory(self):
        return os.path.join(self.get_working_directory(), ENTITY_DIRNAME)

    def get_bin_dir(self):
        return os.path.join(self.get_entity_directory(), BINARIES_DIRNAME)

    def get_json_path(self):
        return os.path.join(self.get_entity_directory(), ENTITY_JSON)

    def add_bin(self,
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
        self.save_binary(binary_id)

    def remove_binary(self, binary_name):
        # Remove the numpy array from class and disk
        p = os.path.join(self.get_working_directory(), BINARIES_DIRNAME, f"{binary_name}.npy")
        os.remove(p)

        # Remove the key in model
        self._binaries_model.pop(binary_name)
        self._binaries.pop(binary_name)
        self.to_disk(self.parent_folder)

    def get_binary(self, binary_name):
        if binary_name not in self._binaries:
            self.load_binary(binary_name)
        return self._binaries[binary_name]

    def get_binary_model(self, binary_name):
        return self._binaries_model[binary_name]

    def get_binary_list_filtered(self, key_filterer: Callable[[Dict], bool]):
        return [i for i in self._binaries_model if key_filterer(self._binaries_model[i])]

    def count_binaries_memory(self) -> int:
        """
        :return: The memory occupied by all the binaries together.
        """
        return sum([self._binaries[k].nbytes for k in self._binaries.keys()])

    def __str__(self):
        res = f"Representation of {self.entity_id}:\n"
        if self.parent_folder is not None:
            res += f"Last saved at {self.get_working_directory()}\n"
        res += f"Loaded Binaries: {list(self._binaries.keys())}\n"
        # for binary in self._binaries.keys():
        #     res += f"{binary}: shape: {self._binaries[binary].shape} (type: {self._binaries[binary].dtype})\n"
        #     res += f"{json.dumps(self._binaries_model[binary], indent=4)}\n"
        res += f'Attrs: {list(self._attrs.keys())}\n'
        res += f'Children: {list(self._children.keys())}\n'
        res += f'Bins: {list(self._binaries_model.keys())}\n'
        return res

    def __repr__(self):
        return str(self)


class Subject(Entity):
    """
    In a medical context a subject is concerned with the data of one patient.
    For example, a patient has classes (disease_1, ...), imaging (US video, CT volumetric data, x-ray image, ...),
    text (symptom description, history) and structured data (date of birth, nationality...).

    Wherever possible the data is saved in json format, but for example for imaging only the metadata is saved
    as json, the actual image file can be found in the binaries-list.

    In future releases a complete changelog will be saved in a format suitable for process mining.
    """

    @classmethod
    def build_empty(cls, name: str):
        res = cls(name=name, model={
            "classes": {}
        })
        return res

    def set_class(self, class_name: str, value: str, for_dataset: Dataset = None, for_binary=""):
        """
        Set a class to true. Classes are stored by their unique string.

        Absence of a class indicates an unknown.

        Hint: If two states of one class can be true to the same time, do not model them as one class.
        Instead of modelling ligament tear as one class, define a binary class for each different ligament.

        :param class_name: Unique string that is used to identify the class.
        :param value: boolean indicating
        :param for_dataset: If provided, set_class checks for compliance with the dataset.
        :param for_binary: If provided, only set the class of the binary and not the whole subject
        :return:
        """
        if for_dataset is not None:
            class_obj = for_dataset.get_class(class_name)
            # print(f"Setting {class_name} to {value}")
            # print(f"{for_dataset.name} tells us about the class:\n{class_obj}")
            if value not in class_obj['values']:
                raise Exception(f"{class_name} cannot be set to {value} according to {for_dataset.entity_id}")

        # Set value
        if for_binary:
            if "classes" not in self._binaries_model[for_binary]["meta_data"]:
                self._binaries_model[for_binary]["meta_data"]["classes"] = {}
            self._binaries_model[for_binary]["meta_data"]["classes"][class_name] = value
        else:
            self.json_model['classes'][class_name] = value

    def get_class_value(self, class_name: str, for_binary=''):
        if for_binary:
            if "classes" not in self._binaries_model[for_binary]["meta_data"]:
                return "--Removed--"
            if class_name in self._binaries_model[for_binary]["meta_data"]["classes"]:
                return self._binaries_model[for_binary]["meta_data"]["classes"][class_name]
        else:
            if class_name in self.json_model['classes']:
                return self.json_model['classes'][class_name]
        return "--Removed--"

    def remove_class(self, class_name: str, for_binary=''):
        if for_binary:
            self._binaries_model[for_binary]["meta_data"]["classes"].pop(class_name)
        else:
            self.json_model['classes'].pop(class_name)

    def get_image_stack_keys(self):
        raise NotImplementedError()
        # return self.get_binary_list_filtered(lambda x: x["binary_type"] == BinaryType.ImageStack.value)

    def add_source_image_by_arr(self,
                                src_im,
                                binary_name: str = "src",
                                structures: (str, str) = None,
                                extra_info: Dict = None):
        """
        Only adds images, not volumes or videos! Unless it is already in shape (frames, width, height, channels).
        Multi-channel images are assumed to be channels last.
        Grayscale images are assumed to be of shape (width, height)
        :param src_im:
        :param binary_name:
        :param structures:
        :param extra_info: Extra info for a human. Must contain only standard types to be json serializable
        :return:
        """
        # Save corresponding json metadata
        meta = {} if structures is None else {"structures": structures}
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
            res = src_im.astype(np.uint8) if src_im.dtype != np.uint8 else src_im
            meta["image_type"] = "video"
        else:
            raise Exception("This array can not be an image, check shape!")

        # Extra info
        if extra_info is not None:
            meta["extra"] = extra_info

        self.add_bin(binary_name, res, b_type=BinaryType.ImageStack.value, meta_data=meta)

    def delete_gt(self, mask_of: str = None, frame_number=0):
        print(f"Deleting ground truth of {mask_of} at frame {frame_number}")
        gt_name = f"gt_{mask_of}_{frame_number}"  # naming convention
        self.remove_binary(gt_name)

    def add_new_gt_by_arr(self,
                          gt_arr: np.ndarray,
                          structure_names: List[str] = None,
                          mask_of: str = None,
                          frame_number=0):
        """

        :param gt_arr:
        :param structure_names:
        :param mask_of:
        :param frame_number:
        :return: The identifier of this binary
        """
        err_msg = "#structures must correspond to the #channels or be 1 in the case of a single indicated structure"
        assert len(gt_arr.shape) == 2 or gt_arr.shape[2] == len(structure_names), err_msg
        assert gt_arr.dtype == np.bool, "Convert to bool, because the ground truth is assumed to be binary!"
        assert mask_of is not None, "Currently for_src can not be inferred, set a value!"

        if len(gt_arr.shape) == 2:
            # This is a single indicated structure without a last dimension, add it!
            gt_arr = np.reshape(gt_arr, (gt_arr.shape[0], gt_arr.shape[1], 1))

        meta = {
            "mask_of": mask_of,
            "frame_number": frame_number,
            "structures": structure_names
        }
        gt_name = f"gt_{mask_of}_{frame_number}"  # naming convention
        self.add_bin(gt_name, gt_arr, b_type=BinaryType.ImageMask.value, meta_data=meta)

        # TODO set class for this binary if a pixel is non zero in the corresponding binary
        return gt_name

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

    def get_masks_of(self, b_name: str, frame_numbers=False):
        res = []
        for m_name in self.get_binary_list_filtered(
                lambda x: x['binary_type'] == BinaryType.ImageMask.value and x['meta_data']['mask_of'] == b_name):
            if frame_numbers:
                res.append(self.get_binary_model(m_name)['meta_data']['frame_number'])
            else:
                res.append(m_name)
        return res

    def get_manual_struct_segmentations(self, struct_name: str) -> Tuple[Dict[str, List[int]], int]:
        res, n = {}, 0

        def filter_imgstack_structs(x: Dict):
            is_img_stack = x['binary_type'] == BinaryType.ImageStack.value
            contains_struct = struct_name in x['meta_data']['structures']
            return is_img_stack and contains_struct

        # Iterate over image stacks that contain the structure
        for b_name in self.get_binary_list_filtered(filter_imgstack_structs):
            # Find the masks of this binary and list them

            bs = self.get_masks_of(b_name)
            n += len(bs)
            if bs:
                res[b_name] = bs

        return res, n


class Dataset(Entity):

    @classmethod
    def build_new(cls, name: str, dir_path: str, example_class=True):
        if os.path.exists(os.path.join(dir_path, name)):
            raise Exception("The directory for this Dataset already exists, use from_disk to load it.")
        res = cls(name, model={
            "subjects": [],
            "splits": {},
            "classes": {},
            "structure_templates": {
                "basic": {"foreground": MaskType.Blob.value,
                          "outline": MaskType.Line.value}
            }
        })
        if example_class:
            res.add_class("example_class", class_type=ClassType.Nominal,
                          values=["Unknown", "Tiger", "Elephant", "Mouse"])
        res.to_disk(dir_path)
        return res

    @classmethod
    def download(cls, url: str, local_path='.', dataset_name: str = None):
        working_dir_path = download_and_extract(url, parent_dir=local_path, dir_name=dataset_name)
        return Dataset.from_disk(working_dir_path)

    def update_weights(self, struct_name: str, weights: np.ndarray):
        print(f"Updating the weights for {struct_name}")
        self.add_bin(struct_name, weights)

    def add_class(self, class_name: str, class_type: ClassType, values: List[str]):
        """
        Adds a class on a dataset level.
        :param class_name:
        :param class_type:
        :param values:
        :return:
        """
        obj = {
            "class_type": class_type.value,
            "values": values
        }
        self.json_model['classes'][class_name] = obj

    def get_class_names(self):
        return list(self.json_model['classes'].keys())

    def get_class(self, class_name: str) -> Union[Dict, None]:
        if class_name in self.json_model['classes']:
            return self.json_model['classes'][class_name]
        else:
            return None

    def remove_class(self, class_name: str):
        self.json_model['classes'].pop(class_name)

    def save_into(self, dir_path: str, properly_formatted=True, vis=True) -> None:
        old_working_dir = self.get_working_directory()
        super().to_disk(dir_path, properly_formatted=properly_formatted)
        for i, te_key in enumerate(self.json_model["subjects"]):
            te_path = os.path.join(old_working_dir, te_key)
            te = Subject.from_disk(te_path)
            te.to_disk(self.get_working_directory())
            if vis:
                sg.OneLineProgressMeter('My Meter', i + 1, len(self.json_model['subjects']), 'key',
                                        f'Subject: {te.entity_id}')

    def get_structure_template_names(self):
        return list(self.json_model["structure_templates"].keys())

    def get_structure_template_by_name(self, tpl_name):
        return self.json_model["structure_templates"][tpl_name]

    def save_subject(self, s: Subject, split=None, auto_save=True):
        """


        :param s: The subject to be saved into this dataset
        :param split:
        :param auto_save: If True, the subject is immediately written to disk
        :return:
        """
        # Add the name of the subject into the model
        if s.entity_id not in self.json_model["subjects"]:
            self.json_model["subjects"].append(s.entity_id)

        # Save it as a child directory to this dataset
        s.to_disk(self.get_working_directory())

        if split is not None:
            self.append_subject_to_split(s, split)

        if auto_save:
            self.to_disk(self.parent_folder)

    def get_subject_name_list(self, split='') -> List[str]:
        """
        Computes the list of subjects in this dataset.
        :param split: Dataset splits of the subjects
        :return: List of the names of the subjects
        """
        if not split:
            subjects = self.json_model["subjects"]
        else:
            subjects = self.json_model["splits"][split]
        return subjects

    def append_subject_to_split(self, s: Subject, split: str):
        # Create the split if it does not exist
        if split not in self.json_model["splits"]:
            self.json_model["splits"][split] = []

        self.json_model["splits"][split].append(s.entity_id)

    def filter_subjects(self, filterer: Callable[[Subject], bool], viz=False) -> List[str]:
        """
        Returns a list with the names of subjects of interest.
        :param filterer: If the filterer returns true, the subject is added to the list
        :param viz: Whether or not a progress meter should be displayed
        :return: The list of subjects of interest
        """
        res: List[str] = []
        for i, s_name in enumerate(self.json_model["subjects"]):
            te = self.get_subject_by_name(s_name)
            if filterer(te):
                res.append(te.entity_id)
            if viz:
                sg.OneLineProgressMeter("Filtering subjects", i + 1,
                                        len(self.json_model['subjects']),
                                        'key',
                                        f'Subject: {te.entity_id}')
        return res

    def delete_subjects(self, del_ls: List[Subject]) -> None:
        """
        Deletes a list of subjects
        :param del_ls: List of instances of subjects
        :return:
        """
        for s in tqdm(del_ls, desc="Deleting subjects"):
            del_name = s.name
            s.delete_on_disk()
            self.json_model["subjects"].remove(del_name)
            for split in self.json_model["splits"]:
                if del_name in self.json_model["splits"][split]:
                    self.json_model["splits"][split].remove(del_name)
        self.to_disk(self.parent_folder)

    def get_subject_by_name(self, s_name: str):
        if s_name not in self.json_model['subjects']:
            raise Exception('This dataset does not contain a subject with this name')
        res = Subject.from_disk(os.path.join(self.get_working_directory(), s_name))
        return res

    def get_summary(self) -> str:
        split_summary = ""
        for split in self.json_model["splits"]:
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

            stacks = te.get_binary_list_filtered(struct_appender)
            return len(stacks) != 0

        self.filter_subjects(lambda x: te_filterer(x))
        return seg_structs

    def get_subject_count(self, split=''):
        if not split:
            return len(self.json_model["subjects"])
        else:
            return len(self.json_model["splits"][split])

    def __len__(self):
        return self.get_subject_count()

    def __getitem__(self, item):
        return self.get_subject_by_name(item)

    def __iter__(self):
        from trainer.ml.data_loading import get_subject_gen
        return get_subject_gen(self)
