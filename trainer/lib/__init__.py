from trainer.lib.enums import MaskType, BinaryType, ClassType, ClassSelectionLevel, BinarySaveProvider
from trainer.lib.JsonClass import JsonClass
from trainer.lib.misc import get_img_from_fig, create_identifier, standalone_foldergrab, make_converter_dict_for_enum, \
    load_grayscale_from_disk, download_and_extract, slugify
from trainer.lib.import_utils import add_imagestack, add_image_folder, import_dicom

try:
    from trainer.lib.demo_dataset import get_test_logits
except ImportError:
    print("Please install all requirements")
