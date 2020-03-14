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
        self.crit = ml.SegCrit(1., 2., (0.5, 0.5))

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
            gt_inv = np.invert(gt[:, :, 0].astype(np.bool) | gt[:, :, 1].astype(np.bool)).astype(np.float32)
            # gt = gt[:, :, 0].astype(np.float32)
            # gt = cv2.resize(gt, (384, 384))
            # # gt = np.expand_dims(gt, 0)
            gt_stacked = np.zeros((3, 384, 384), dtype=np.int)
            gt_stacked[0, :, :] = cv2.resize(gt_inv, (384, 384))
            gt_stacked[0, :, :] = cv2.resize(gt, (384, 384))
            return [([im], gt_stacked)]
        return [([im], np.empty(0))]
