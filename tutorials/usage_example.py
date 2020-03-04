import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import trainer.ml as ml
import trainer.lib as lib
from trainer.ml.data_loading import get_subject_gen, get_img_mask_pair


def visualize_one_train_pair(ds: ml.Dataset, image_stack_index=1, segmentation_name="femur", frame=0) -> plt.Figure:
    """
    For demonstration, just visualize some content of one subject
    """
    g = get_subject_gen(ds)
    one_subject = next(g)
    videos = one_subject.get_image_stack_keys()
    print(f"All image stacks associated with {one_subject.name}:\n{videos}\n")
    first_video_name = videos[image_stack_index]

    frame, mask = get_img_mask_pair(
        one_subject,
        first_video_name,
        segmentation_name,
        frame_number=frame)

    pair_fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(frame)
    sns.heatmap(mask, ax=ax2)
    return pair_fig


if __name__ == '__main__':
    # A dataset can be loaded as following, using a direct path to a zipfile:
    ds = ml.Dataset.download(url='https://rwth-aachen.sciebo.de/s/1qO95mdEjhoUBMf/download',
                             local_path='./data',  # Your local data folder
                             dataset_name='crucial_ligament_diagnosis'  # Name of the dataset
                             )

    fig = visualize_one_train_pair(ds)
    fig.show()
    # If you want to export some information of the dataset
    # you can iterate through the subjects in deterministic order
    # as following:
    export_dir = './export'
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
        while os.path.exists(export_dir):
            pass
    os.mkdir(export_dir)
    for s in ds:
        # All information about the subject is now capsuled in s
        print(s.name)
        subject_folder = os.path.join(export_dir, s.name)
        os.mkdir(subject_folder)
        for im_name in s.get_image_stack_keys():
            imstack_folder = os.path.join(subject_folder, im_name)
            os.mkdir(imstack_folder)
            imstack = s.get_binary(im_name)
            np.save(os.path.join(imstack_folder, f'im.npy'), imstack)
            for gt_name in s.get_masks_of(im_name):
                gt = s.get_binary(gt_name)
                gt_model = s.get_binary_model(gt_name)
                f_number = str(gt_model['meta_data']['frame_number'])
                np.save(os.path.join(imstack_folder, f'{f_number}.npy'), gt)
