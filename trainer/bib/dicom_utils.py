import pydicom
from typing import List

ALLOWED_TYPES = [
    str,
    List[str],
    int,
    float
]


def import_dicom(dicom_path: str):
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
