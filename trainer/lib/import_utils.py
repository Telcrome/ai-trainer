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


def add_image_folder(split: lib.Split, folder_path: str, progress=True, sess=lib.Session()) -> None:
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
    :param sess: database session, defaults to a new session
    """
    top_level_files = os.listdir(folder_path)
    for i, file_name in enumerate(top_level_files):
        if progress:
            sg.OneLineProgressMeter(
                title=f'Adding Image Folder',
                key='key',
                current_value=i + 1,
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
                s_existing = sess.query(lib.Subject).filter(lib.Subject.name == p_id_clean).first()
                if s_existing is not None:
                    print("load patient")
                    s = s_existing
                else:
                    print("Create new patient")
                    s = lib.Subject.build_new(p_id_clean)
                im = lib.ImStack.build_new(img_data)
                s.ims.append(im)
                split.sbjts.append(s)
                sess.add(s)
            else:  # Everything else is assumed to be a traditional image file
                # Create the new subject
                raise NotImplementedError()


def import_subject(split: lib.Split, subject_path: str, semsegtpl: lib.SemSegTpl):
    s_name = os.path.split(subject_path)[-1]
    s = lib.Subject.build_new(s_name)
    split.sbjts.append(s)
    imstack_paths = [os.path.join(subject_path, p) for p in os.listdir(subject_path)]
    for imstack_path in imstack_paths:
        gts_paths = [(os.path.join(imstack_path, p), os.path.splitext(p)[0].replace('us_bone', '')) for p in
                     os.listdir(imstack_path) if p != 'im.npy']
        im_arr = np.load(os.path.join(imstack_path, 'im.npy'))
        imstack = lib.ImStack.build_new(src_im=im_arr)
        s.ims.append(imstack)
        for gt_path, f_number in gts_paths:
            gt_arr = np.load(gt_path)
            imstack.add_ss_mask(gt_arr, semsegtpl, for_frame=f_number)


def add_import_folder(split: lib.Split, folder_path: str, semsegtpl: lib.SemSegTpl):
    subject_paths = [os.path.join(folder_path, fn) for fn in os.listdir(folder_path)]
    for sp in subject_paths:
        print(f'Importing {sp}')
        import_subject(split, sp, semsegtpl)


def export_to_folder(split: lib.Split, folder_path: str):
    os.mkdir(folder_path)
    for s in split.sbjts:
        # All information about the subject is now capsuled in s
        print(s.name)
        subject_folder = os.path.join(folder_path, s.name)
        os.mkdir(subject_folder)
        for i, im in enumerate(s.ims):
            imstack_folder = os.path.join(subject_folder, f"{i}")
            os.mkdir(imstack_folder)
            np.save(os.path.join(imstack_folder, f'im.npy'), im.values())
            for gt in im.semseg_masks:
                np.save(os.path.join(imstack_folder, f'{gt.tpl.name}{gt.for_frame}.npy'), gt.values())
