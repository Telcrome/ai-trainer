from trainer.ml.data_model import Dataset, Subject

try:  # The following modules depend on torch, for CI systems without torch a try block is required
    from trainer.ml.data_loading import random_struct_generator, random_subject_generator
    from trainer.ml.utils import *
    from trainer.ml.visualization import VisBoard, LogWriter
    from trainer.ml import seg_network
    from trainer.ml.losses import dice_loss, FocalLoss
except ImportError:
    print("Please install all requirements")
