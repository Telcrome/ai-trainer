# %%

from typing import Iterable

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import trainer.ml as ml
import trainer.lib as lib
from trainer.ml.data_loading import get_subject_gen, random_struct_generator, get_img_mask_pair
from trainer.ml import batcherize, resize, normalize_im


def g_convert(g):
    for vid, gt, f in g:
        res = np.zeros((vid.shape[1], vid.shape[2], 3))
        if f >= 2:
            # Discard most of the video, deeplab can handle only 3 frames
            res[:, :, 0] = vid[f - 2, :, :, 0]
            res[:, :, 1] = vid[f - 1, :, :, 0]
            res[:, :, 2] = vid[f, :, :, 0]
        else:
            res[:, :, 0] = vid[f, :, :, 0]
            res[:, :, 1] = vid[f, :, :, 0]
            res[:, :, 2] = vid[f, :, :, 0]
        gt_stacked = np.zeros((gt.shape[0], gt.shape[1], 2), dtype=np.float32)
        gt_stacked[:, :, 0] = gt.astype(np.float32)
        gt_stacked[:, :, 1] = np.invert(gt).astype(gt_stacked.dtype)
        yield normalize_im(res).astype(np.float32), gt_stacked
        
def vis(g: Iterable):
    im, gt = next(g)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    im_2d = im[0, :, :, 0]
    gt_2d = gt[0, :, :, 0]
    sns.heatmap(im_2d, ax=ax1)
    sns.heatmap(gt_2d, ax=ax2)
    fig.show()


def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    dice = ml.dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

if __name__ == '__main__':
    # ds = ml.Dataset.download(url='https://rwth-aachen.sciebo.de/s/1qO95mdEjhoUBMf/download',
    #                          local_path='./data',  # Your local data folder
    #                          dataset_name='crucial_ligament_diagnosis'  # Name of the dataset
    #                          )
    ds = ml.Dataset.from_disk('./data/b8_old_ultrasound_segmentation')

    structure_name = 'gt'  # The structure that we now train for
    bce_weight = 0.5
    BATCH_SIZE = 4

    # Simple generator preprocessing chain
    g_raw = random_struct_generator(ds, structure_name, split='train')
    g_extracted = g_convert(g_raw)
    g_resized = resize(g_extracted, (384, 384))
    g_train = ml.channels_last_to_first(batcherize(g_resized, batchsize=BATCH_SIZE))

    vis(g_train)

    device = torch.device('cuda')
    model = ml.seg_network.ResNetUNet(n_class=2)
    model = model.to(device)
    model.train()
    N = ds.get_subject_count(split='train')

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    visboard = ml.VisBoard(run_name=lib.create_identifier('test'))

    def run_epoch(epoch: int):
        print(f'Starting epoch: {epoch} with {N} training examples')
        epoch_loss_sum = 0.
        with torch.no_grad():
            # Visualize model output
            x, y = next(g_train)
            y_ = model(torch.from_numpy(x).to(device))
            for i in range(y_.shape[0]):
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                sns.heatmap(x[i, 0, :, :], ax=ax1)
                sns.heatmap(y[i, 0, :, :], ax=ax2)
                sns.heatmap(y_.cpu().numpy()[i, 0, :, :], ax=ax3)
                visboard.add_figure(fig, group_name=f'Before Epoch{epoch}')

        for i in tqdm(range(N // BATCH_SIZE)):
            x, y = next(g_train)
            x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

            optimizer.zero_grad()

            outputs = model(x)
            bce = F.binary_cross_entropy_with_logits(outputs, y)

            outputs = torch.sigmoid(outputs)
            dice = ml.dice_loss(outputs, y)

            loss = bce * bce_weight + dice * (1 - bce_weight)

            loss.backward()
            optimizer.step()
            # scheduler.step()
            # metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
            # metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
            # metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
            epoch_loss_sum += loss.data.cpu().numpy() * y.size(0)
            epoch_loss = epoch_loss_sum / (i+1)
            visboard.writer.add_scalar(f'loss/train epoch {epoch+1}', epoch_loss, i)
        print(f"Epoch result: {epoch_loss_sum / N}")
        
    for epoch in range(20):
        run_epoch(epoch)

    # for x, y in g_train:
    #     print(x.shape)
    #     print(y.shape)
    #     break
