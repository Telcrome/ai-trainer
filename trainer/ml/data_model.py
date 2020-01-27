"""
The data model aims to simplify machine learning on complex data structures.
For example classifying a subject (medical patient) by both its gender and between 1 and 4 ultrasound videos.

A dataset contains
- Subjects (Which are the training examples)
- Model Weights
- Config Json files
"""

from __future__ import annotations  # Important for function annotations of symbols that are not loaded yet

import os
from typing import Callable, List, Dict, Set, Tuple, Union

import PySimpleGUI as sg
import numpy as np
from tqdm import tqdm

from trainer.lib import JsonClass, download_and_extract, create_identifier
from trainer.lib import MaskType, BinaryType, ClassType


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
                                structures: Dict[str, str] = None,
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
            raise Exception("This image can not be an image, check shape!")

        # Extra info
        if extra_info is not None:
            meta["extra"] = extra_info

        self.add_binary(binary_name, res, b_type=BinaryType.ImageStack.value, meta_data=meta)

    def add_file_as_imagestack(self,
                               file_path: str,
                               binary_name='',
                               structures: Dict[str, str] = None):
        """
        Takes an image path and tries to deduce the type of image from the path ending.
        No path ending is assumed to be a DICOM file (not a DICOM folder)
        """
        if not binary_name:
            binary_name = create_identifier(hint="binary")

        file_ending = os.path.splitext(file_path)[1]
        if file_ending in ['', '.dcm']:
            from trainer.ml import append_dicom_to_subject
            append_dicom_to_subject(self.get_working_directory(), file_path, binary_name=binary_name,
                                    seg_structs=structures)
        elif file_ending == '.b8':
            from trainer.lib.misc import load_b8
            img_data = load_b8(file_path)
            self.add_source_image_by_arr(
                img_data,
                binary_name=binary_name,
                structures=structures
            )
        elif file_ending in ['.mp4']:
            print('Video!')
        else:
            raise Exception('This file type is not understood')

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

    def stack_gts(self, src_name):
        """

        :param src_name: Source binary name
        :return:
        """
        im_shape = self.get_binary(src_name).shape

        def is_gt_of_im(gt: Dict) -> bool:
            if "mask_of" in gt["meta_data"]:
                return gt["meta_data"]["mask_of"] == src_name

        gt_binaries = self.get_binary_list_filtered(is_gt_of_im)
        # print(len(gt_binaries))
        assert len(gt_binaries) == 1  # For a single image there should only be one ground truth
        # res = np.zeros((im_shape[0], im_shape[1], len(gt_binaries) - 1))
        # for channel_index, b_name in enumerate(gt_binaries):
        #     res[:, :, channel_index] = self.get_binary(b_name)
        return self.get_binary(gt_binaries[0])

    def get_grayimage_training_tuple_raw(self, src_name: str):
        training_input = self.get_binary(src_name)
        training_output = self.stack_gts(src_name)
        return training_input[0, :, :, 0].astype(np.float32), training_output.astype(np.float32)

    def get_manual_struct_segmentations(self, struct_name: str) -> Tuple[Dict[str, List[int]], int]:
        res, N = {}, 0

        def filter_imgstack_structs(x: Dict):
            is_img_stack = x['binary_type'] == BinaryType.ImageStack.value
            contains_struct = struct_name in x['meta_data']['structures']
            return is_img_stack and contains_struct

        # Iterate over image stacks that contain the structure
        for b_name in self.get_binary_list_filtered(filter_imgstack_structs):
            # Find the masks of this binary and list them
            bs = []
            for m_name in self.get_binary_list_filtered(
                    lambda x: x['binary_type'] == BinaryType.ImageMask.value and x['meta_data']['mask_of'] == b_name):
                bs.append(m_name)
                N += 1

            if bs:
                res[b_name] = bs

        return res, N


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

    def get_subject_name_list(self, split=None) -> List[str]:
        """
        Computes the list of subjects in this dataset.
        :param split: Dataset splits of the subjects
        :return: List of the names of the subjects
        """
        if split is None:
            subjects = self.json_model["subjects"]
        else:
            subjects = self.json_model["splits"][split]
        return subjects

    def append_subject_to_split(self, s: Subject, split: str):
        # Create the split if it does not exist
        if split not in self.json_model["splits"]:
            self.json_model["splits"][split] = []

        self.json_model["splits"][split].append(s.name)

    def append_dataset(self, d_path: str, source_split: str = None, target_split: str = None) -> List:
        """
        Appends the structures from d to this dataset.
        Then copies the subjects.
        :param d_path: Path of the appended dataset
        :param source_split: The split in d that the subjects come from.
        :param target_split: The split that the subjects are appended to.
        :return: The names of the subjects that are copied from d to self
        """
        raise NotImplementedError()  # TODO Update this method
        # Load the other dataset from disk
        d = Dataset.from_disk(d_path)

        # Append structure templates
        self._json_model["structure_templates"].extend(d._json_model["structure_templates"])
        self._json_model["structure_templates"] = list(set(self._json_model["structure_templates"]))

        # Add the subjects
        copied = []
        if source_split is None:
            for s_name in tqdm(d._json_model["subjects"], desc=f"Adding {d.name}"):
                copied.append(s_name)
                self.save_subject(d.get_subject_by_name(s_name), split=target_split, auto_save=False)
        else:
            for s_name in tqdm(d._json_model["splits"][source_split], desc=f"Adding {d.name}"):
                copied.append(s_name)
                self.save_subject(d.get_subject_by_name(s_name), split=target_split, auto_save=False)
        self.to_disk(self._last_used_parent_dir)
        return copied

    def add_image_folder(self, parent_folder: str, structures: Dict[str, str], split=None, progress=True):
        """
        Iterates through a folder.

        If a file is found, a new subject is created with only that file.
        If a directory is found, a new subject is created with all files that live within that directory.
        If a dicom file is found, the image is appended to the subject with that patient_id

        Supported file formats:
        - DICOM (no extension or .dcm)
        - Standard image files
        - B8 files (.b8)

        :param parent_folder: Top level folder path
        :param structures: Dict with structure_name: structure_type pairs
        :param split: The dataset split this data is appended to.
        :param progress: If true, displays a progress bar
        :return:
        """
        top_level_files = os.listdir(parent_folder)
        for i, file_name in enumerate(top_level_files):
            if progress:
                sg.OneLineProgressMeter(
                    title=f'Adding Image Folder',
                    key='key',
                    current_value=i,
                    max_value=len(top_level_files),
                    grab_anywhere=True,
                )

            if os.path.isdir(os.path.join(parent_folder, file_name)):
                # Create the new subject
                s_name = os.path.splitext(file_name)[0]
                s = Subject.build_empty(s_name)
                self.save_subject(s, split=split, auto_save=False)
                dir_children = os.listdir(os.path.join(parent_folder, file_name))

                for image_path in dir_children:
                    # Append this image to the subject
                    s.add_file_as_imagestack(
                        os.path.join(parent_folder, file_name, image_path),
                        binary_name=os.path.splitext(image_path)[0],
                        structures=structures,
                    )
                self.save_subject(s, split=split, auto_save=False)
            else:  # Assume this is a file
                file_ext = os.path.splitext(os.path.join(parent_folder, file_name))[1]
                if file_ext in ['', '.dcm']:  # Most likely a dicom file
                    from trainer.lib.dicom_utils import import_dicom
                    img_data, meta = import_dicom(os.path.join(parent_folder, file_name))
                    from trainer.lib import slugify
                    p_id = meta['PatientID']
                    p_id_clean = slugify(p_id)
                    if p_id_clean in self.get_subject_name_list():
                        print("load patient")
                        s = Subject.from_disk(os.path.join(self.get_working_directory(), p_id_clean))
                    else:
                        print("Create new patient")
                        s = Subject.build_empty(p_id_clean)
                    s.add_source_image_by_arr(img_data,
                                              create_identifier(hint='DICOM'),
                                              structures=structures,
                                              extra_info=meta)
                    self.save_subject(s, split=split, auto_save=False)
                else:  # Everything else is assumed to be a traditional image file
                    # Create the new subject
                    s_name = os.path.splitext(file_name)[0]
                    s = Subject.build_empty(s_name)

                    s.add_file_as_imagestack(
                        os.path.join(parent_folder, file_name),
                        binary_name=os.path.splitext(file_name)[0],
                        structures=structures,
                    )
                    self.save_subject(s, split=split, auto_save=False)

        self.to_disk()

    def add_ml_folder(self, folder: str, split=None) -> None:
        """
        Assumes a folder structure of the following form:

        - subject 1
            - im (training images)
            - gt1 (segmentation maps for class gt1)
            - gt2 (segmentation maps for class gt2)
            - ...
        - subject 2
            - ...

        The name of the source image and its corresponding ground truths must be identical
        :param split: The training split (train, test...) that the folder is appended to
        :param folder: The path to the parent folder
        :return:
        """
        raise NotImplementedError()  # Update this method
        structure_names = [item for item in os.listdir(folder) if
                           item not in ['im'] and os.path.isdir(os.path.join(folder, item))]
        source_folder = os.path.join(folder, 'im')
        if not os.path.exists(source_folder):
            raise FileNotFoundError("Directory doesnt contain source images")

        source_paths = os.listdir(source_folder)
        for src_path in source_paths:
            te = Subject.build_with_src_image(os.path.splitext(src_path)[0],
                                              os.path.join(source_folder, src_path))
            for structure_name in structure_names:
                structure_path = os.path.join(folder, structure_name)
                gt_p = os.path.join(structure_path, f"{te.name}{os.path.splitext(src_path)[1]}")
                te.add_new_gt_by_path(structure_name, gt_p)
            self.save_subject(te, split=split, auto_save=False)

        self.to_disk()

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
            split_summary += f"""{split}: {self.__len__(split=split)}\n"""
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

    def get_subject_count(self, split=None):
        if split is None:
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
    ds_path = download_and_extract(
        online_url='https://rwth-aachen.sciebo.de/s/1qO95mdEjhoUBMf/download',
        parent_dir='./data',  # Your local data folder
        dir_name='crucial_ligament_diagnosis'  # Name of the dataset
    )
    ds = Dataset.from_disk(ds_path)
