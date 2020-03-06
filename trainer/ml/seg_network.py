import random
from typing import Tuple, List

import cv2
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from torchvision import models

import trainer.lib as lib
import trainer.ml as ml


class SegCrit(nn.Module):
    """
    Criterion which is optimized for semantic segmentation tasks.

    Expects targets in the range between 0 and 1 and the logits of the predictions.

    >>> import trainer.ml as ml
    >>> import trainer.lib as lib
    >>> import numpy as np
    >>> import torch
    >>> np.random.seed(0)
    >>> alpha, beta, loss_weights = 1., 2., (0.5, 0.5)
    >>> sc = ml.SegCrit(alpha, beta, loss_weights)
    >>> preds, target = lib.get_test_logits(shape=(8, 1, 3, 3)), np.random.randint(size=(8, 1, 3, 3), low=0, high=2)
    >>> preds, target = torch.from_numpy(preds.astype(np.float32)), torch.from_numpy(target.astype(np.float32))
    >>> sc.forward(preds, target)
    tensor(6.8156)
    """

    def __init__(self, alpha, beta, loss_weights: Tuple):
        super().__init__()
        self.loss_weights = loss_weights
        self.focal_loss = ml.FocalLoss(alpha=alpha, gamma=beta, logits=True)

    def forward(self, logits, target):
        bce = self.focal_loss(logits, target)
        outputs = torch.sigmoid(logits)
        dice = ml.dice_loss(outputs, target)
        return bce * self.loss_weights[0] + dice * self.loss_weights[1]


class WrapperNet(nn.Module):

    def __init__(self, wrapped_net: nn.Module):
        super().__init__()
        self.wrapped_net = wrapped_net

    def forward(self, inps):
        return self.wrapped_net(inps[0])


class SegNetwork:

    def __init__(self):
        # model = ResNetUNet(n_class=n_classes)
        self.in_channels, self.n_classes = 3, 2
        pan = smp.PAN(in_channels=self.in_channels, classes=self.n_classes)
        # pan.load_state_dict(torch.load(r'C:\Users\rapha\Desktop\epoch78.pt'))
        self.model = WrapperNet(pan)
        self.opti = optim.Adam(self.model.parameters(), lr=5e-3)
        self.crit = SegCrit(1., 2., (0.5, 0.5))

    @staticmethod
    def visualize_input_batch(te: Tuple[np.ndarray, np.ndarray]) -> None:
        x, y = te
        for batch_id in range(x.size()[0]):
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
            sns.heatmap(x[batch_id, 0, :, :], ax=ax1)
            sns.heatmap(x[batch_id, 1, :, :], ax=ax2)
            sns.heatmap(y[batch_id, 0, :, :], ax=ax3)
            sns.heatmap(y[batch_id, 1, :, :], ax=ax4)
            fig.show()

    @staticmethod
    def preprocess_segmap(s: lib.Subject,
                          mode: ml.ModelMode = ml.ModelMode.Train) -> List[Tuple[List[np.ndarray], np.ndarray]]:
        imstack_with_masks = list(filter(lambda istck: len(istck.semseg_masks) > 0, s.ims))
        imstack: lib.ImStack = random.choice(imstack_with_masks)

        if not mode == ml.ModelMode.Usage:
            mask: lib.SemSegMask = random.choice(imstack.semseg_masks)
            gt = mask.get_ndarray()
        else:
            raise NotImplementedError()
            # mask = random.randint(0, imstack.get_ndarray().shape[0])
        im = imstack.get_ndarray()[mask.for_frame]
        # im = cv2.cvtColor(imstack.get_ndarray()[mask.for_frame], cv2.COLOR_GRAY2RGB)

        if mode == ml.ModelMode.Train:
            # Augmentation
            seq = iaa.Sequential([
                iaa.Dropout([0.01, 0.2]),  # drop 5% or 20% of all pixels
                iaa.Crop(percent=(0, 0.1)),
                iaa.Fliplr(0.5),
                iaa.Sharpen((0.0, 1.0)),  # sharpen the image
                # iaa.SaltAndPepper(0.1),
                iaa.WithColorspace(
                    to_colorspace="HSV",
                    from_colorspace="RGB",
                    children=iaa.WithChannels(
                        0,
                        iaa.Add((0, 50))
                    )
                ),
                iaa.Sometimes(p=0.5, then_list=[iaa.Affine(rotate=(-10, 10))])
                # iaa.ElasticTransformation(alpha=50, sigma=5)  # apply water effect (affects segmaps)
            ], random_order=True)
            segmap = SegmentationMapsOnImage(gt, shape=im.shape)
            im, gt = seq(image=im, segmentation_maps=segmap)
            gt = gt.arr

        # Processing
        im = cv2.resize(im, (384, 384))
        im = np.rollaxis(ml.normalize_im(im), 2, 0)

        if not mode == ml.ModelMode.Usage:
            gt_inv = np.invert(gt[:, :, 0].astype(np.bool)).astype(np.float32)
            gt = gt[:, :, 0].astype(np.float32)
            # gt = cv2.resize(gt, (384, 384))
            # # gt = np.expand_dims(gt, 0)
            gt_stacked = np.zeros((2, 384, 384), dtype=np.float32)
            gt_stacked[0, :, :] = cv2.resize(gt, (384, 384))
            gt_stacked[1, :, :] = cv2.resize(gt_inv, (384, 384))
            return [([im], gt_stacked)]
        return [([im], np.empty(0))]


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

        # self.apply(torch.nn.init.xavier_normal)

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
