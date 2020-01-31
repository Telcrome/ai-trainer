# %%

import random
from typing import Iterable, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import trainer.lib as lib
import trainer.ml as ml
from trainer.ml import normalize_im
from trainer.ml.data_loading import get_mask_for_frame, random_subject_generator


def subject_processor(s: ml.Subject) -> Tuple[np.ndarray, np.ndarray]:
    is_names = s.get_image_stack_keys()
    is_name = random.choice(is_names)
    available_structures = s.get_structure_list(image_stack_key=is_name)
    selected_struct = random.choice(list(available_structures.keys()))
    im = s.get_binary(is_name)
    selected_frame = random.randint(0, im.shape[0] - 1)
    gt = get_mask_for_frame(s, is_name, selected_struct, selected_frame)

    # Processing
    im = cv2.resize(im[selected_frame], (384, 384))
    im = np.rollaxis(normalize_im(cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)), 2, 0)
    gt = gt.astype(np.float32)
    gt = cv2.resize(gt, (384, 384))
    # gt = np.expand_dims(gt, 0)
    gt_stacked = np.zeros((2, gt.shape[0], gt.shape[1]), dtype=np.float32)
    gt_stacked[0, :, :] = gt.astype(np.float32)
    gt_stacked[1, :, :] = np.invert(gt.astype(np.bool)).astype(gt_stacked.dtype)

    return im, gt_stacked


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
    loss_weights = (0.1, 1.5)
    BATCH_SIZE = 4
    criterion = ml.FocalLoss(alpha=1., gamma=2., logits=False)
    EPOCHS = 60

    # Simple generator preprocessing chain
    g_subjects = random_subject_generator(ds, subject_processor, split='train', batchsize=BATCH_SIZE)

    device = torch.device('cuda')
    model = ml.seg_network.ResNetUNet(n_class=2)
    model = model.to(device)
    model.train()
    N = ds.get_subject_count(split='train')

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    visboard = ml.VisBoard(run_name=lib.create_identifier('test'))


    def run_epoch(epoch: int):
        print(f'Starting epoch: {epoch} with {N} training examples')
        epoch_loss_sum = 0.
        with torch.no_grad():
            # Visualize model output
            x, y = next(g_subjects)
            y_ = torch.sigmoid(model(torch.from_numpy(x).to(device)))
            for i in range(y_.shape[0]):
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
                sns.heatmap(x[i, 0, :, :], ax=ax1)
                sns.heatmap(y[i, 0, :, :], ax=ax2)
                sns.heatmap(y_.cpu().numpy()[i, 0, :, :], ax=ax3)
                sns.heatmap((y_.cpu().numpy()[i, 0, :, :] > 0.5).astype(np.int8), ax=ax4)
                visboard.add_figure(fig, group_name=f'Before Epoch{epoch}')

        for i in tqdm(range(N // BATCH_SIZE)):
            # x, y = next(g_train)
            x, y = next(g_subjects)
            x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

            optimizer.zero_grad()

            outputs = model(x)

            outputs = torch.sigmoid(outputs)
            bce = criterion(outputs, y)
            dice = ml.dice_loss(outputs, y)

            # bce_weight = bce_weight - 1. / EPOCHS
            loss = bce * loss_weights[0] + dice * loss_weights[1]

            loss.backward()
            optimizer.step()
            # scheduler.step()
            # metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
            # metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
            # metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
            epoch_loss_sum += loss.data.cpu().numpy() * y.size(0)
            epoch_loss = epoch_loss_sum / (i + 1)
            visboard.add_scalar(f'loss/train epoch {epoch + 1}', epoch_loss, i)
        print(f"Epoch result: {epoch_loss_sum / N}")


    for epoch in range(10):
        run_epoch(epoch)

    # for x, y in g_train:
    #     print(x.shape)
    #     print(y.shape)
    #     break
