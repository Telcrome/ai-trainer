import matplotlib.pyplot as plt
import seaborn as sns
import imgaug.augmenters as iaa

import trainer
import trainer.ml as ml


def visualize_generator(g_vis, name="Gen") -> plt.Figure:
    im, gt, f = next(g_vis)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    sns.heatmap(im.squeeze(), ax=ax1)
    sns.heatmap(gt, ax=ax2)
    return fig


if __name__ == '__main__':
    # Load the Dataset we want to perform augmentations on
    ds_path, _ = trainer.bib.standalone_foldergrab(folder_not_file=True)
    ds = ml.Dataset.from_disk(ds_path)
    board = ml.VisBoard()

    example_structure = list(ds.compute_segmentation_structures().keys())[0]
    g = trainer.ml.data_loading.random_struct_generator(ds, struct_name=example_structure)
    g_aug = ml.utils.pair_augmentation(g, [
        # iaa.Crop(px=(1, 16), keep_size=False),
        iaa.Fliplr(0.5),
        # iaa.GammaContrast((0.5, 1.5)),
        # iaa.Sometimes(0.5,
        #               iaa.GaussianBlur(sigma=(0, 3.0))
        #               ),
    ])

    board.add_figure(visualize_generator(g, name='raw'), group_name='raw')
    board.add_figure(visualize_generator(g_aug, name='augmented'), group_name='aug')
    board.add_figure(visualize_generator(g, name='raw'), group_name='raw')
    board.add_figure(visualize_generator(g_aug, name='augmented'), group_name='aug')
    board.add_figure(visualize_generator(g, name='raw'), group_name='raw')
    board.add_figure(visualize_generator(g_aug, name='augmented'), group_name='aug')
