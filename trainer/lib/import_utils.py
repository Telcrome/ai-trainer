"""
Collection of utility function which can be used to add content from other file-formats and sources
to the convenient trainer dataset-format.
"""
import os
from typing import Dict, List, Tuple

import numpy as np
import PySimpleGUI as sg
import imageio

import trainer.ml as ml
import trainer.lib as lib

ALLOWED_TYPES = [
    str,
    List[str],
    int,
    float
]


def add_imagestack(s: ml.Subject, file_path: str, binary_id='', structures: Dict[str, str] = None) -> None:
    """
    Takes an image path and tries to deduce the type of image from the path ending.
    No path ending is assumed to be a DICOM file (not a DICOM folder)
    """
    if not binary_id:
        binary_id = lib.create_identifier(hint="binary")

    file_ending = os.path.splitext(file_path)[1]
    if file_ending in ['', '.dcm']:
        from trainer.ml import append_dicom_to_subject
        append_dicom_to_subject(s.get_working_directory(), file_path, binary_name=binary_id,
                                seg_structs=structures)
    elif file_ending == '.b8':
        from trainer.lib.misc import load_b8
        img_data = load_b8(file_path)
        s.add_source_image_by_arr(
            img_data,
            binary_name=binary_id,
            structures=structures
        )
    elif file_ending in ['.jpg', '.png']:
        img_data = imageio.imread(file_path)
        s.add_source_image_by_arr(
            img_data,
            binary_name=binary_id,
            structures=structures
        )
    elif file_ending in ['.mp4']:
        print('Video!')
    else:
        raise Exception('This file type is not understood')


def import_dicom(dicom_path: str):
    import pydicom
    ds = pydicom.dcmread(dicom_path)
    ds.decompress()

    if 'PixelData' in ds:

        img_data = ds.pixel_array

        meta = {}
        for key in ds.keys():
            v = ds[key]
            try:  # Use try to check if v is even subscriptable
                key, val = v.keyword, v.value

                if type(val) in ALLOWED_TYPES:
                    if key and val:  # Checks for empty strings
                        meta[key] = val
                        print(f"Added {key} with value {val}")
                    else:
                        print(f"{key} seems to be empty, skip!")
                else:
                    print(f"{key} has the wrong type: {type(val)}")
            except Exception as e:
                print(e)
                print(f"ignored: {str(v)}")
    else:
        raise Exception("The dicom file seems to not contain any pixel data")

    return img_data, meta


def add_image_folder(ds: ml.Dataset, parent_folder: str, structures: Dict[str, str], split=None, progress=True):
    """
    Iterates through a folder.

    If a file is found, a new subject is created with only that file.
    If a directory is found, a new subject is created with all files that live within that directory.
    If a dicom file is found, the image is appended to the subject with that patient_id

    Supported file formats:
    - DICOM (no extension or .dcm)
    - Standard image files
    - B8 files (.b8)

    :param ds: A trainer.ml.Dataset that this folder will be appended to
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
            s = ml.Subject.build_empty(s_name)
            ds.save_subject(s, split=split, auto_save=False)
            dir_children = os.listdir(os.path.join(parent_folder, file_name))

            ds.save_subject(s, split=split, auto_save=False)
            for image_path in dir_children:
                # Append this image to the subject
                s.add_file_as_imagestack(
                    os.path.join(parent_folder, file_name, image_path),
                    binary_name=os.path.splitext(image_path)[0],
                    structures=structures,
                )
                s.to_disk()
        else:  # Assume this is a file
            file_ext = os.path.splitext(os.path.join(parent_folder, file_name))[1]
            if file_ext in ['', '.dcm']:  # Most likely a dicom file
                img_data, meta = import_dicom(os.path.join(parent_folder, file_name))
                from trainer.lib import slugify
                p_id = meta['PatientID']
                p_id_clean = slugify(p_id)
                if p_id_clean in ds.get_subject_name_list():
                    print("load patient")
                    s = ml.Subject.from_disk(os.path.join(ds.get_working_directory(), p_id_clean))
                else:
                    print("Create new patient")
                    s = ml.Subject.build_empty(p_id_clean)
                ds.save_subject(s, split=split, auto_save=False)
                s.add_source_image_by_arr(img_data,
                                          lib.create_identifier(hint='DICOM'),
                                          structures=structures,
                                          extra_info=meta)
                s.to_disk()
            else:  # Everything else is assumed to be a traditional image file
                # Create the new subject
                s_name = os.path.splitext(file_name)[0]
                s = ml.Subject.build_empty(s_name)

                ds.save_subject(s, split=split, auto_save=False)
                add_imagestack(
                    s,
                    os.path.join(parent_folder, file_name),
                    binary_id=os.path.splitext(file_name)[0],
                    structures=structures,
                )
                s.to_disk()

    ds.to_disk()


def append_subject(ds: ml.Dataset,
                   im_path: Tuple[str, str],
                   gt_paths: List[Tuple[str, str]],
                   seg_structs: Dict[str, str],
                   split='') -> None:
    """
    Appends one subject with an image and corresponding masks to a dataset split.

    TODO: Add support for directories to add subjects with multiple images with corresponding gts
    """
    im_path, im_name = im_path
    s = ml.Subject.build_empty(name=im_name)
    ds.save_subject(s, split=split, auto_save=False)

    add_imagestack(s, im_path, binary_id=im_name, structures=seg_structs)
    im_arr = s.get_binary(im_name)

    if gt_paths:
        gt_arr = np.zeros((im_arr.shape[1], im_arr.shape[2], len(gt_paths)), dtype=np.bool)
        for i, (gt_path, gt_name) in enumerate(gt_paths):
            arr = imageio.imread(os.path.join(gt_path, im_name))
            gt_arr[:, :, i] = arr
        s.add_new_gt_by_arr(gt_arr, structure_names=[v for (_, v) in gt_paths], mask_of=im_name, frame_number=0)

    # print(f'File path: {im_path} with name: {im_name}')

    s.to_disk()
