from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils import data
from tqdm import tqdm

import trainer.lib as lib
import trainer.ml as ml
from trainer.ml.torch_utils import device
from trainer.ml.data_loading import get_mask_for_frame, random_subject_generator

if __name__ == '__main__':
    # ds = ml.Dataset.download(url='https://rwth-aachen.sciebo.de/s/1qO95mdEjhoUBMf/download',
    #                          local_path='./data',  # Your local data folder
    #                          dataset_name='crucial_ligament_diagnosis'  # Name of the dataset
    #                          )
    ds = ml.Dataset.from_disk('./data/full_ultrasound_seg_0_0_9')

    structure_name = 'gt'  # The structure that we now train for
    loss_weights = (0.1, 1.5)
    BATCH_SIZE = 4
    EPOCHS = 60

    N = ds.get_subject_count(split='train')

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    seg_network = ml.seg_network.SegNetwork("ResNet_UNet", 3, 2, ds, batch_size=BATCH_SIZE)
    visboard = ml.VisBoard(run_name=lib.create_identifier('test'))

    train_loader = data.DataLoader(seg_network.get_torch_dataset(split='train'), batch_size=BATCH_SIZE, num_workers=4)
    machine_loader = data.DataLoader(seg_network.get_torch_dataset(split='machine'), batch_size=BATCH_SIZE)

    def run_epoch(epoch: int):
        print(f'Starting epoch: {epoch} with {N} training examples')
        epoch_loss_sum = 0.
        with torch.no_grad():
            # Visualize model output
            x, y = next(iter(machine_loader))
            x, y = x.to(device), y.to(device)
            y_ = torch.sigmoid(seg_network.model(x))
            for i in range(y_.shape[0]):
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
                sns.heatmap(x.cpu().numpy()[i, 0, :, :], ax=ax1)
                sns.heatmap(y.cpu().numpy()[i, 0, :, :], ax=ax2)
                sns.heatmap(y_.cpu().numpy()[i, 0, :, :], ax=ax3)
                sns.heatmap((y_.cpu().numpy()[i, 0, :, :] > 0.5).astype(np.int8), ax=ax4)
                visboard.add_figure(fig, group_name=f'Before Epoch{epoch}')

        for i, (x, y) in tqdm(enumerate(train_loader)):
            # x, y = seg_network.sample_minibatch(split='train')
            x, y = x.to(device), y.to(device)

            loss = seg_network.train_on_minibatch((x, y))

            epoch_loss_sum += loss
            epoch_loss = epoch_loss_sum / (i + 1)
            visboard.add_scalar(f'loss/train epoch {epoch + 1}', epoch_loss, i)
        print(f"Epoch result: {epoch_loss_sum / N}")


    for epoch in range(10):
        run_epoch(epoch)

    # for x, y in g_train:
    #     print(x.shape)
    #     print(y.shape)
    #     break
