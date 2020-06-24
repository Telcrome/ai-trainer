"""
Machine Learning Package
------------------------

The machine learning package bundles everything directly related to learning from the data.
For internal models, torch is tightly integrated into trainer.
"""
try:  # The following modules depend on torch, for CI systems without torch a try block is required
    from trainer.ml.utils import distance_transformed, pair_augmentation, normalize_im, one_hot_to_cont, \
        cont_to_ont_hot, reduce_by_attention, insert_np_at, duplicate_columns, pad, split_into_regions
    from trainer.ml.losses import dice_loss, FocalLoss, SegCrit
    from trainer.ml.layers import ConvGRUCell, ResidualBlock
    from trainer.ml.torch_utils import ModelMode, InMemoryDataset, device as torch_device, TrainerMetric, AccuracyMetric
except ImportError as e:
    print(f"Please install all requirements: {e}")
