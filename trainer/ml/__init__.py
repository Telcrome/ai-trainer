"""
Machine Learning Package
------------------------

The machine learning package bundles everything directly related to learning from the data.
For internal models, torch is tightly integrated into trainer.
"""
try:  # The following modules depend on torch, for CI systems without torch a try block is required
    from trainer.ml.utils import distance_transformed, pair_augmentation, normalize_im, one_hot_to_cont, \
        reduce_by_attention
    from trainer.ml.visualization import VisBoard, LogWriter, logger
    from trainer.ml.losses import dice_loss, FocalLoss, SegCrit
    from trainer.ml.layers import ConvGRUCell, ResidualBlock
    from trainer.ml.torch_utils import ModelMode, ModelTrainer, InMemoryDataset, device as torch_device, \
        TrainerMetric, AccuracyMetric, SemSegDataset
    from trainer.ml.seg_network import SegNetwork
except ImportError as e:
    print(f"Please install all requirements: {e}")
