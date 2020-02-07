import os
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import cv2
import imageio
import numpy as np
import seaborn as sns
import torch
from torch.utils import data

import trainer.lib as lib
import trainer.ml as ml
from trainer.ml.torch_utils import device, ModelMode


def train_model():
    for epoch in range(50):
        with torch.no_grad():
            # Visualize model output
            for fig in seg_network.visualize_prediction(machine_loader):
                fig.suptitle("B8 Stuff")
                visboard.add_figure(fig, group_name=f'Before Epoch{epoch}, B8')
            for fig in seg_network.visualize_prediction(test_loader):
                fig.suptitle("Test Set")
                visboard.add_figure(fig, group_name=f'Before Epoch{epoch}, Test')
            # for fig in seg_network.visualize_prediction(train_loader):
            #     fig.suptitle("Train Set")
            #     visboard.add_figure(fig, group_name=f'Before Epoch{epoch}, Train')
        seg_network.run_epoch(train_loader, epoch, N, batch_size=BATCH_SIZE)

        # Save model weights
        torch.save(seg_network.model.state_dict(), f'./model_weights/epoch{epoch}.pt')


def save_predictions(dir_path: str, split='machine'):
    p_dir_path = os.path.join(dir_path, split)
    if not os.path.exists(p_dir_path):
        os.mkdir(p_dir_path)
    seg_network.model.eval()
    ss = ds.get_subject_name_list(split=split)
    for s_name in ss:
        s = ds.get_subject_by_name(s_name)
        is_names = s.get_image_stack_keys()
        print(is_names)
        for is_name in is_names:
            im = s.get_binary(is_name)
            print(f'Exporting {is_name}')
            for frame_i in range(im.shape[0]):
                frame = im[frame_i, :, :, 0]
                imsize = frame.shape[1], frame.shape[0]
                imageio.imwrite(os.path.join(p_dir_path, f'{is_name}_{frame_i}_im.png'), frame)
                frame_prep = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                frame_prep = cv2.resize(frame_prep, (384, 384))
                frame_prep = np.rollaxis(ml.normalize_im(frame_prep), 2, 0)
                frame_batch = np.array([frame_prep] * BATCH_SIZE)
                frame_batch = torch.from_numpy(frame_batch)
                pred_batch = torch.sigmoid(seg_network.model(frame_batch.to(device)))
                pred = pred_batch.detach().cpu().numpy()[0, 0]
                pred = (pred * 255.).astype(np.uint8)
                pred = cv2.resize(pred, imsize)
                imageio.imwrite(os.path.join(p_dir_path, f'{is_name}_{frame_i}_pred.png'), pred)


if __name__ == '__main__':
    # ds = ml.Dataset.download(url='https://rwth-aachen.sciebo.de/s/1qO95mdEjhoUBMf/download',
    #                          local_path='./data',  # Your local data folder
    #                          dataset_name='crucial_ligament_diagnosis'  # Name of the dataset
    #                          )
    ds = ml.Dataset.from_disk(r'C:\Users\rapha\Desktop\data\old_b8')

    structure_name = 'gt'  # The structure that we now train for

    BATCH_SIZE = 4
    EPOCHS = 60

    N = ds.get_subject_count(split='train')

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    visboard = ml.VisBoard(run_name=lib.create_identifier('test'))
    seg_network = ml.seg_network.SegNetwork("SegNetwork", 3, 2, ds, batch_size=BATCH_SIZE, vis_board=visboard)
    # seg_network.model.load_state_dict(torch.load(f'./model_weights/epoch{28}.pt'))
    from trainer.ml.torch_utils import get_capacity

    print(f"Capacity of the network: {get_capacity(seg_network.model)}")

    train_loader = data.DataLoader(
        seg_network.get_torch_dataset(split='train', mode=ModelMode.Train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4)
    test_loader = data.DataLoader(seg_network.get_torch_dataset(split='test', mode=ModelMode.Eval),
                                  batch_size=BATCH_SIZE, shuffle=True)
    machine_loader = data.DataLoader(seg_network.get_torch_dataset(split='machine', mode=ModelMode.Usage),
                                     batch_size=BATCH_SIZE,
                                     shuffle=True)
    # for x, y in g_train:
    #     print(x.shape)
    #     print(y.shape)
    #     break
    train_model()
    # save_predictions('./out2/', split='train')
    # save_predictions('./out2/', split='test')
    # save_predictions('./out2/', split='machine')
