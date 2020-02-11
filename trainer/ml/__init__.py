"""
Machine Learning Package
------------------------

The machine learning package bundles everything directly related to learning from the data.
For internal models, torch is tightly integrated into trainer.
"""
try:  # The following modules depend on torch, for CI systems without torch a try block is required
    from trainer.ml.data_loading import image_stack_classification_preprocessor
    from trainer.ml.utils import *
    from trainer.ml.visualization import VisBoard, LogWriter
    from trainer.ml.losses import dice_loss, FocalLoss
    from trainer.ml.torch_utils import ModelMode, TrainerModel, TorchDataset, device as torch_device
    from trainer.ml.seg_network import SegNetwork, SegCrit
except ImportError:
    print("Please install all requirements")
