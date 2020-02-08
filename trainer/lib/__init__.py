from trainer.lib.data_api import JsonClass, Subject, Dataset, MaskType, BinaryType, ClassType, ClassSelectionLevel, \
    BinarySaveProvider, dir_is_json_class
from trainer.lib.import_utils import add_imagestack, add_image_folder, import_dicom
from trainer.lib.misc import get_img_from_fig, create_identifier, standalone_foldergrab, make_converter_dict_for_enum, \
    load_grayscale_from_disk, slugify, download_and_extract

try:
    from trainer.lib.demo_data import get_test_logits, get_test_jsonclass
except ImportError:
    print("Please install all requirements")
