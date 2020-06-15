from trainer.lib.config import config, BIG_BIN_KEY, DB_CON_KEY
from trainer.lib.data_model import Session, NumpyBinary, ClassType, ClassDefinition, Classifiable, MaskType, \
    SemSegClass, SemSegTpl, SemSegMask, ImStack, Subject, Split, Dataset, reset_data_model, sbjts_splits_association, \
    DataModelBase, engine
from trainer.lib.import_utils import add_imagestack, add_image_folder, import_dicom, add_import_folder, export_to_folder
from trainer.lib.misc import get_img_from_fig, create_identifier, standalone_foldergrab, make_converter_dict_for_enum, \
    load_grayscale_from_disk, slugify, download_and_extract, delete_dir, pick_from_list
from trainer.lib.gen_utils import product, sample_randomly
from trainer.lib.grammar import Grammar, RULE
from trainer.lib.DslSemantics import DslSemantics
from trainer.lib.logging import LogWriter, logger, Experiment, ExperimentResult, LogBase

# Initialize database
DataModelBase.metadata.create_all(engine)
LogBase.metadata.create_all(engine)


def reset_complete_database():
    reset_data_model()

    log_tables = [Experiment, ExperimentResult, ]

    DataModelBase.metadata.drop_all(bind=engine, tables=[c.__table__ for c in log_tables])

    DataModelBase.metadata.create_all(engine)
    LogBase.metadata.create_all(engine)
