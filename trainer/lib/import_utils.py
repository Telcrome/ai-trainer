"""
Collection of utility function which can be used to add content from other file-formats and sources
to the convenient trainer dataset-format.
"""
import os
from typing import Dict, List, Tuple

import PySimpleGUI as sg
import imageio
import numpy as np

import trainer.lib as lib

DICOM_ALLOWED_TYPES = [
    str,
    List[str],
    int,
    float
]


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

                if type(val) in DICOM_ALLOWED_TYPES:
                    if key and val:  # Checks for empty strings
                        meta[key] = val
                        # print(f"Added {key} with value {val}")
                    else:
                        pass
                        # print(f"{key} seems to be empty, skip!")
                else:
                    pass
                    # print(f"{key} has the wrong type: {type(val)}")
            except Exception as e:
                print(e)
                print(f"ignored: {str(v)}")
    else:
        raise Exception("The dicom file seems to not contain any pixel data")

    return img_data, meta


def add_imagestack(s: lib.Subject, file_path: str, binary_id='') -> None:
    """
    Takes an image path and tries to deduce the type of image from the path ending.
    No path ending is assumed to be a DICOM file (not a DICOM folder)
    """
    if not binary_id:
        binary_id = lib.create_identifier(hint="binary")

    file_ending = os.path.splitext(file_path)[1]
    if file_ending in ['', '.dcm']:
        img_data, meta = import_dicom(file_path)
        im = lib.ImStack.build_new(src_im=img_data, extra_info=meta)
        s.ims.append(im)
    elif file_ending == '.b8':
        from trainer.lib.misc import load_b8
        img_data = load_b8(file_path)
        im = lib.ImStack.build_new(img_data)
        s.ims.append(im)
    elif file_ending in ['.jpg', '.png']:
        img_data = imageio.imread(file_path)
        im = lib.ImStack.build_new(img_data)
        s.ims.append(im)
    elif file_ending in ['.mp4']:
        print('Video!')
    else:
        raise Exception('This file type is not understood')


def add_image_folder(split: lib.Split, folder_path: str, progress=True) -> None:
    """
    Iterates through a folder and adds its contents to a dataset.

    If a file is found, a new subject is created with only that file.
    If a directory is found, a new subject is created with all files that live within that directory.
    If a dicom file is found, the image is appended to the subject with that patient_id

    Supported file formats:
    - DICOM (no extension or .dcm)
    - Standard image files
    - B8 files (.b8)

    :param folder_path: Top level folder path
    :param split: The dataset split this data is appended to.
    :param progress: If true, displays a progress bar
    """
    top_level_files = os.listdir(folder_path)
    for i, file_name in enumerate(top_level_files):
        if progress:
            sg.OneLineProgressMeter(
                title=f'Adding Image Folder',
                key='key',
                current_value=i,
                max_value=len(top_level_files),
                grab_anywhere=True,
            )

        if os.path.isdir(os.path.join(folder_path, file_name)):
            raise NotImplementedError()
        else:  # Assume this is a file
            file_ext = os.path.splitext(os.path.join(folder_path, file_name))[1]
            if file_ext in ['', '.dcm']:  # Most likely a dicom file
                img_data, meta = import_dicom(os.path.join(folder_path, file_name))
                from trainer.lib import slugify
                p_id = meta['PatientID']
                p_id_clean = slugify(p_id)
                s_existing = lib.Session().query(lib.Subject).filter(lib.Subject.name == p_id_clean).first()
                if s_existing is not None:
                    print("load patient")
                    s = s_existing
                else:
                    print("Create new patient")
                    s = lib.Subject.build_new(p_id_clean)
                s.ims.append(lib.ImStack.build_new(img_data))
                split.sbjts.append(s)
            else:  # Everything else is assumed to be a traditional image file
                # Create the new subject
                raise NotImplementedError()

    lib.Session().commit()


def append_subject(ds: lib.Dataset,
                   im_path: Tuple[str, str],
                   gt_paths: List[Tuple[str, str]],
                   seg_structs: Dict[str, str],
                   split='',
                   artefact_threshold=150) -> None:
    """
    Appends one subject with an image and corresponding masks to a dataset split.

    TODO: Add support for adding subjects with multiple images with corresponding gts
    :param ds:
    :param im_path:
    :param gt_paths:
    :param seg_structs:
    :param split:
    :param artefact_threshold: Threshold for removing artifacts. Pass none if no thresholding should happen.
    """
    im_path, im_file = im_path
    im_name, im_ext = os.path.splitext(im_file)
    s = lib.Subject.build_empty(name=im_name)
    ds.save_subject(s, split=split, auto_save=False)

    add_imagestack(s, im_path, binary_id=im_name, structures=seg_structs)
    im_arr = s._get_binary(im_name)

    if gt_paths:
        gt_arr = np.zeros((im_arr.shape[1], im_arr.shape[2], len(gt_paths)), dtype=np.bool)
        for i, (gt_path, gt_name) in enumerate(gt_paths):
            arr = imageio.imread(os.path.join(gt_path, im_file))
            if artefact_threshold is not None:
                arr = arr > artefact_threshold
            gt_arr[:, :, i] = arr
        s.add_sem_seg(gt_arr, structure_names=[v for (_, v) in gt_paths], mask_of=im_name, frame_number=0)

    # print(f'File path: {im_path} with name: {im_name}')

    s.to_disk()
