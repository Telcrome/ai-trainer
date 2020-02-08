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
from typing import Dict, Callable, Union, Any, List, Tuple, Set

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


BINARIES_DIRNAME = "binaries"
JSON_MODEL_FILENAME = "json_model.json"
BINARY_TYPE_KEY = "binary_type"
PICKLE_EXT = 'pickle'


def build_full_filepath(name: str, dir_path: str):
    basename = os.path.splitext(name)[0]  # If a file ending is provided it is ignored
    return os.path.join(dir_path, f"{basename}.json")


def dir_is_json_class(dir_name: str, json_checker: Callable[[str], bool] = None):
    if json_checker is None:
        def json_checker(json_val: str) -> bool:
            return True
    # Check if the json exists:
    json_path = os.path.join(dir_name, JSON_MODEL_FILENAME)
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            json_content = json.load(f)
        if json_checker(json_content):
            return True
    # TODO Check other constraints (binaries)
    return False


class JsonClass:
    """
    Intended to be subclassed by classes which need to persist their state
    in the form of numpy binaries (Video, images...) and json metadata (name, attributes...).
    """

    def __init__(self, name: str, model: Dict = None, b_model: Dict = None):
        self.name = name
        self.binaries_dir_path = None

        if model is not None:
            self.json_model = model
        else:
            self.json_model = {}

        if b_model is not None:
            self._binaries_model = b_model
        else:
            self._binaries_model = {}
        self._binaries: Dict[str, np.ndarray] = {}

        self._last_used_parent_dir = None

    @classmethod
    def from_disk(cls, dir_path: str, pre_load_binaries=False):
        if not dir_is_json_class(dir_path):
            raise Exception(f"{dir_path} is not a valid directory.")

        name = os.path.basename(dir_path)

        full_file_path = os.path.join(dir_path, JSON_MODEL_FILENAME)
        with open(full_file_path, 'r') as f:
            json_content = json.load(f)
        res = cls(name, json_content["payload"], json_content["binaries"])
        res._last_used_parent_dir = os.path.dirname(dir_path)

        # Load binaries
        res.binaries_dir_path = os.path.join(dir_path, BINARIES_DIRNAME)
        if pre_load_binaries:
            binaries_paths_ls = os.listdir(res.binaries_dir_path)
            for binary_path in binaries_paths_ls:
                binary_name = os.path.splitext(os.path.basename(binary_path))[0]
                res.load_binary(binary_name)
        return res

    def load_binary(self, binary_id) -> None:
        path_no_ext = os.path.join(self.binaries_dir_path, binary_id)
        if self.get_binary_provider(binary_id) == BinarySaveProvider.Pickle:
            with open(f'{path_no_ext}.{PICKLE_EXT}', 'rb') as f:
                binary_payload = pickle.load(f)
        else:
            binary_payload = np.load(f'{path_no_ext}.npy', allow_pickle=False)
        self._binaries[binary_id] = binary_payload

    def to_disk(self, dir_path: str = "", properly_formatted=True, prompt_user=False) -> None:
        if not dir_path:
            dir_path = self._last_used_parent_dir

        # Create the parent directory for this instance
        dir_path = os.path.join(dir_path, self.name)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        file_name = os.path.join(dir_path, JSON_MODEL_FILENAME)
        self._last_used_parent_dir = os.path.dirname(dir_path)

        # Write the json model file
        with open(file_name, 'w+') as f:
            save_json = {
                "payload": self.json_model,
                "binaries": self._binaries_model
            }
            if properly_formatted:
                json.dump(save_json, f, indent=4)
            else:
                json.dump(save_json, f)

        # Write all binaries
        self.binaries_dir_path = os.path.join(dir_path, BINARIES_DIRNAME)
        if not os.path.exists(self.binaries_dir_path):
            os.mkdir(self.binaries_dir_path)
        for binary_key in self._binaries:
            self.save_binary(binary_key)

        if prompt_user:
            os.startfile(file_name)

    def get_binary_provider(self, binary_key: str):
        return BinaryType.provider_map()[self._binaries_model[binary_key][BINARY_TYPE_KEY]]

    def save_binary(self, binary_key) -> None:
        """
        Writes the selected binary on disk.
        :param binary_key: ID of the binary
        """
        if self._last_used_parent_dir is None:
            raise Exception(f" Before saving {binary_key}, save {self.name} to disk!")
        if self.binaries_dir_path is None:
            self.binaries_dir_path = os.path.join(self._last_used_parent_dir, BINARIES_DIRNAME)
        path_no_ext = os.path.join(self.binaries_dir_path, binary_key)
        if self.get_binary_provider(binary_key) == BinarySaveProvider.Pickle:
            with open(f'{path_no_ext}.{PICKLE_EXT}', 'wb') as f:
                pickle.dump(self._binaries[binary_key], f)
        else:
            np.save(path_no_ext, self._binaries[binary_key])

    def delete_on_disk(self, blocking=True):
        shutil.rmtree(self.get_working_directory(), ignore_errors=True)
        if blocking:
            while os.path.exists(self.get_working_directory()):
                pass

    def get_parent_directory(self):
        return self._last_used_parent_dir

    def get_working_directory(self):
        return os.path.join(self._last_used_parent_dir, self.name)

    def get_json_path(self):
        return os.path.join(self.get_working_directory(), JSON_MODEL_FILENAME)

    def add_binary(self,
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
        self.to_disk(self._last_used_parent_dir)

    def get_binary(self, binary_name):
        if binary_name not in self._binaries:
            self.load_binary(binary_name)
        return self._binaries[binary_name]

    def get_binary_model(self, binary_name):
        return self._binaries_model[binary_name]

    def get_binary_list_filtered(self, key_filterer: Callable[[Dict], bool]):
        return [i for i in self._binaries_model if key_filterer(self._binaries_model[i])]

    def get_image_stack_keys(self):
        return self.get_binary_list_filtered(lambda x: x["binary_type"] == BinaryType.ImageStack.value)

    def count_binaries_memory(self) -> int:
        """
        :return: The memory occupied by all the binaries together.
        """
        return sum([self._binaries[k].nbytes for k in self._binaries.keys()])

    def __str__(self):
        res = f"Representation of {self.name}:\n"
        if self._last_used_parent_dir is not None:
            res += f"Last saved at {self.get_working_directory()}\n"
        res += f"Binaries: {len(self._binaries.keys())}\n"
        for binary in self._binaries.keys():
            res += f"{binary}: shape: {self._binaries[binary].shape} (type: {self._binaries[binary].dtype})\n"
            res += f"{json.dumps(self._binaries_model[binary], indent=4)}\n"
        res += json.dumps(self.json_model, indent=4)
        return res

    def __repr__(self):
        return str(self)


class Subject(JsonClass):
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
                raise Exception(f"{class_name} cannot be set to {value} according to {for_dataset.name}")

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

        self.add_binary(binary_name, res, b_type=BinaryType.ImageStack.value, meta_data=meta)

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
        self.add_binary(gt_name, gt_arr, b_type=BinaryType.ImageMask.value, meta_data=meta)

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


class Dataset(JsonClass):

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
        self.add_binary(struct_name, weights)

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

    def save_into(self, dir_path: str, properly_formatted=True, prompt_user=False, vis=True) -> None:
        old_working_dir = self.get_working_directory()
        super().to_disk(dir_path, properly_formatted=properly_formatted, prompt_user=prompt_user)
        for i, te_key in enumerate(self.json_model["subjects"]):
            te_path = os.path.join(old_working_dir, te_key)
            te = Subject.from_disk(te_path)
            te.to_disk(self.get_working_directory())
            if vis:
                sg.OneLineProgressMeter('My Meter', i + 1, len(self.json_model['subjects']), 'key',
                                        f'Subject: {te.name}')

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
        if s.name not in self.json_model["subjects"]:
            self.json_model["subjects"].append(s.name)

        # Save it as a child directory to this dataset
        s.to_disk(self.get_working_directory())

        if split is not None:
            self.append_subject_to_split(s, split)

        if auto_save:
            self.to_disk(self._last_used_parent_dir)

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

        self.json_model["splits"][split].append(s.name)

    def filter_subjects(self, filterer: Callable[[Subject], bool], viz=False) -> List[str]:
        """
        Returns a list with the names of subjects of interest.
        :param filterer: If the filterer returns true, the subject is added to the list
        :return: The list of subjects of interest
        """
        res: List[str] = []
        for i, s_name in enumerate(self.json_model["subjects"]):
            te = self.get_subject_by_name(s_name)
            if filterer(te):
                res.append(te.name)
            if viz:
                sg.OneLineProgressMeter("Filtering subjects", i + 1,
                                        len(self.json_model['subjects']),
                                        'key',
                                        f'Subject: {te.name}')
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
        self.to_disk(self._last_used_parent_dir)

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
                        if te.name not in seg_structs[structure]:
                            seg_structs[structure] = seg_structs[structure] | {te.name}
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


if __name__ == '__main__':
    parent_folder = 'C:\\Users\\rapha\\Desktop'
    read = False
    if read:
        example_class = JsonClass.from_disk(os.path.join(parent_folder, "jsonclass_example"))
    else:
        example_class = JsonClass("jsonclass_example", {})
        example_class.json_model["test_number"] = 4
        example_class.json_model["test_list"] = [1, 2, 3]
        example_class.json_model["test_dict"] = {"1": "3", "4": "8"}
        example_class.add_binary("test_image", np.ones((100, 100)))
        example_class.to_disk(parent_folder, prompt_user=True)
