import random
from typing import Tuple, Union, Any, Callable

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torchvision import models
from torch.optim import optimizer
import torch.optim as optim

import trainer.ml as ml
from trainer.ml.data_loading import get_mask_for_frame
from trainer.ml.torch_utils import TrainerModel, device


class SegCrit(nn.Module):

    def __init__(self, alpha, beta, loss_weights: Tuple):
        super().__init__()
        self.loss_weights = loss_weights
        self.focal_loss = ml.FocalLoss(alpha=alpha, gamma=beta, logits=False)

    def forward(self, logits, target):
        outputs = torch.sigmoid(logits)
        bce = self.focal_loss(outputs, target)
        dice = ml.dice_loss(outputs, target)
        return bce * self.loss_weights[0] + dice * self.loss_weights[1]


class SegNetwork(TrainerModel):

    def __init__(self,
                 model_name: str,
                 in_channels: int,
                 n_classes: int,
                 ds: ml.Dataset,
                 batch_size=4):
        model = ResNetUNet(n_class=n_classes)
        opti = optim.Adam(model.parameters(), lr=1e-3)
        crit = SegCrit(1., 2., (0.1, 1.5))
        super().__init__(model_name, model, opti, crit, ds, batch_size=batch_size)
        self.in_channels, self.n_classes = in_channels, n_classes

    def visualize_input_batch(self) -> plt.Figure:
        x, y = next(self.gen)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        im_2d = x[0, 0, :, :]
        gt_2d = y[0, 0, :, :]
        sns.heatmap(im_2d, ax=ax1)
        sns.heatmap(gt_2d, ax=ax2)
        return fig

    @staticmethod
    def preprocess(s: ml.Subject) -> Tuple[np.ndarray, np.ndarray]:
        is_names = s.get_image_stack_keys()
        is_name = random.choice(is_names)
        available_structures = s.get_structure_list(image_stack_key=is_name)
        selected_struct = random.choice(list(available_structures.keys()))
        im = s.get_binary(is_name)
        possible_frames = s.get_masks_of(is_name, frame_numbers=True)
        selected_frame = random.choice(possible_frames)
        gt = get_mask_for_frame(s, is_name, selected_struct, selected_frame)

        # Processing
        im = cv2.resize(im[selected_frame], (384, 384))
        im = np.rollaxis(ml.normalize_im(cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)), 2, 0)
        gt = gt.astype(np.float32)
        gt = cv2.resize(gt, (384, 384))
        # gt = np.expand_dims(gt, 0)
        gt_stacked = np.zeros((2, gt.shape[0], gt.shape[1]), dtype=np.float32)
        gt_stacked[0, :, :] = gt.astype(np.float32)
        gt_stacked[1, :, :] = np.invert(gt.astype(np.bool)).astype(gt_stacked.dtype)

        return im, gt_stacked


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(nn.Module):
    def __init__(self, n_class=2):
        super().__init__()

        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)
        self.activation_layer = nn.Softmax2d()

    def forward(self, x: torch.Tensor):
        x_original = self.conv_original_size0(x)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)
        return out
