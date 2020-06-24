import io
import os
import re
import tempfile
import urllib
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Any, TypeVar

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PySimpleGUI as sg


def get_img_from_fig(fig: plt.figure, dpi=180) -> np.ndarray:
    """
    Converts a matplotlib figure into a numpy array.

    :param fig: A matplotlib figure
    :param dpi: image quality, higher is better and takes longer
    :return: np.ndarray with the image content
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()

    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def create_identifier(hint: str = '') -> str:
    """
    Can be used to create unique names for files by exploiting the uniqueness of the current date.
    Be aware that if two identifiers are created during the same second they are equal!
    Follows the form YYYY_MM_DD__hh_mm_ss.

    :return: YYYY_MM_DD__hh_mm_ss_{hint}
    """
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d__%H_%M_%S")
    return f"{dt_string}_{hint}" if hint else dt_string


T = TypeVar('T')


def pick_from_list(ls: List[T], title='Title', rows=-1, columns=-1) -> T:
    """
    Spawns a small gui which allows the user to select from a list.

    Be aware that this works only if a $Display is set
    """
    if rows == -1:
        rows = len(ls) + 1
    if columns == -1:
        columns = max([len(str(x)) for x in ls])
    layout = [[sg.Listbox(values=ls, size=(columns, rows))],
              [sg.Submit()]]
    window = sg.Window(title, layout)
    event, values = window.read()
    window.close()
    return values[0][0]


def delete_dir(dir_path: str, blocking=True, verbose=True):
    if os.path.exists(dir_path):
        if verbose:
            print(f"Deleting {len(os.listdir(dir_path))} binaries from {dir_path}")
        import shutil
        shutil.rmtree(dir_path, ignore_errors=True)
        if blocking:
            while os.path.exists(dir_path):
                pass
    else:
        if verbose:
            print(f"f{dir_path} already did not exist on disk")


def standalone_foldergrab(
        folder_not_file: bool = False,
        optional_inputs: List[Tuple[str, str]] = None,
        optional_choices: List[Tuple[str, str, List[str]]] = None,
        title="Select a folder") -> Tuple[str, Dict[str, str]]:
    """
    Uses a simple blocking GUI for prompting the user for a file or folder path.
    Optionally allows to prompt for additional text inputs as well.

    :param folder_not_file: if True asks for a folder, if False asks for a file
    :param optional_inputs: A list of (description, key) pairs
    :param optional_choices: A list of (description, key, list of options) pairs
    :param title:
    :return: Tuple with path as first entry and a dictionary with the optional text inputs as second entry
    """
    if optional_inputs is None:
        optional_inputs = []
    if optional_choices is None:
        optional_choices = []
    layout = [[sg.Text(text=title)],
              [sg.Input(key='folder_name'), sg.FolderBrowse() if folder_not_file else sg.FileBrowse()]]
    for t in optional_inputs:
        new_item = [sg.Text(text=t[0], size=(15, 1)), sg.InputText(key=t[1])]
        layout.append(new_item)
    for t in optional_choices:
        new_item = [sg.Text(text=t[0], size=(15, 1)), sg.Combo(t[2], key=t[1])]
        layout.append(new_item)
    layout.append([sg.Submit(), sg.Cancel()])

    p = sg.Window(title, layout)
    e, v = p.Read()
    if e is None or e == 'Cancel':
        p.close()
        return "", {}
    elif e == 'Submit':
        p.close()
        res = {t[1]: v[t[1]] for t in optional_inputs}
        res.update({t[1]: v[t[1]] for t in optional_choices})
        return v['folder_name'], res


def make_converter_dict_for_enum(e):
    res = {v: v.value for v in e}
    res.update({v.value: v for v in e})
    return res


def load_grayscale_from_disk(path: str) -> np.ndarray:
    im = imageio.imread(path)
    if len(im.shape) > 2:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    return im


def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    res = re.sub(r'[\\/*?:"<>|]', "", value)
    res = res.replace('\n', '_')
    return res


def load_b8(file_path: str) -> np.ndarray:
    """
    Loads b8 file used by some ultrasound machines.

    :param: file_path: The direct path to the b8 file
    :return: Numpy array with the image data
    """
    with open(file_path, 'r') as f:
        data = np.fromfile(f, dtype=np.uint8)

    last_header_byte = 4 * 19
    raw_header, im = data[:last_header_byte], data[last_header_byte:]

    header = raw_header.view(dtype=np.int32)
    d, w, h = header[1], header[2], header[3]

    res = np.rot90(im.reshape((d, w, h)), k=3, axes=(1, 2))

    # Add an empty channel axis to match the imagestack array format
    res = np.expand_dims(res, axis=3)

    return res


def download_and_extract(online_url: str, parent_dir='./', dir_name: str = None) -> str:
    """
    Can be used to download and extract a .zip dataset file hosted online.
    Assumes the zip to be one directory.
    :param online_url: The url that points directly to a .zip file containing folders with the dataset files
    :param parent_dir: The directory that is used to store the temporary zip and the final extracted folder
    :param dir_name: if provided, the function checks if the directory already exists and does not download it again
    :return: The absolute local path to the directory
    """
    # Check for existence
    if dir_name is not None:
        abs_dirname = os.path.join(parent_dir, dir_name)
        if os.path.exists(abs_dirname):
            print(f"The local copy already exists: {online_url} at {abs_dirname}")
            return abs_dirname
    else:
        print(f"The local copy does not yet exist, proceed to download it")

    # Download
    print(f"Starting downloading from {online_url}")
    temp_file = tempfile.NamedTemporaryFile(mode='w')
    temp_file.close()
    # noinspection PyUnresolvedReferences
    urllib.request.urlretrieve(online_url, temp_file.name)

    # Extract
    print(f"Start extracting {temp_file.name}")
    zf = zipfile.ZipFile(temp_file.name)
    zf.extractall(path=parent_dir)
    zf.close()

    first_filename = zf.filelist[0].filename
    res = os.path.join(parent_dir, str(Path(first_filename).parent))
    if dir_name is not None and dir_name not in res:
        raise Exception(f"The provided directory name {dir_name} did not match the actual directory name {res}")
    return res
