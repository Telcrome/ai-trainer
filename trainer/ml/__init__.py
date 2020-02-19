"""
Machine Learning Package
------------------------

The machine learning package bundles everything directly related to learning from the data.
For internal models, torch is tightly integrated into trainer.
"""
try:  # The following modules depend on torch, for CI systems without torch a try block is required
    from trainer.ml.utils import distance_transformed, im_one_hot, pair_augmentation, normalize_im
    from trainer.ml.visualization import VisBoard, LogWriter, logger
    from trainer.ml.losses import dice_loss, FocalLoss
    from trainer.ml.layers import ConvGRUCell, ConvGRU
    from trainer.ml.torch_utils import ModelMode, ModelTrainer, InMemoryDataset, device as torch_device, \
        TrainerMetric, AccuracyMetric
    from trainer.ml.seg_network import SegNetwork, SegCrit
except ImportError as e:
    print(f"Please install all requirements: {e}")
