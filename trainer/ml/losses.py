from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits: torch.FloatTensor, true: torch.LongTensor, eps=1e-7):
    """
    Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.

    :param logits: a tensor of shape [B, C, H, W]. Corresponds to
        the raw output or logits of the model.
    :param true: a tensor of shape [B, 1, H, W].
    :param eps: added to the denominator for numerical stability.
    :returns dice_loss: the Sørensen–Dice loss.
    """
    if len(true.size()) < 4:
        # Assumption: The empty class dimension is missing
        true = true.unsqueeze(1)

    num_classes = logits.shape[1]
    true = true.long()
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return 1 - dice_loss


class FocalLoss(nn.Module):
    """
    Loss based on cross entropy that emphasizes those outputs that have a large difference to the targets.

    Focal loss is a simple trick which can be used to train networks when class imbalance is present.

    Focusing parameter gamma: Increase to emphasize hard examples and put less effort into optimizing easy ones.
    """
    def __init__(self, alpha=1., gamma=2., logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha, self.gamma, self.logits, self.reduce = alpha, gamma, logits, reduce
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        # targets = targets.long()
        if self.logits:
            bce_loss = self.ce_loss(inputs, targets)
        else:
            bce_loss = F.cross_entropy(inputs, targets, reduce=None)
        pt = torch.exp(-bce_loss)
        f_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduce:
            return torch.mean(f_loss)
        else:
            return f_loss


class SegCrit(nn.Module):
    """
    Criterion which is optimized for semantic segmentation tasks.

    Expects targets in the range between 0 and #C with shape [#B, W, H]
    The logits of the predictions for each class: [#B, #C, W, H].

    >>> import trainer.ml as ml
    >>> import trainer.lib as lib
    >>> import trainer.demo_data as dd
    >>> import numpy as np
    >>> import torch
    >>> np.random.seed(0)
    >>> alpha, beta, loss_weights = 1., 2., (0.5, 0.5)
    >>> sc = ml.SegCrit(alpha, beta, loss_weights)
    >>> preds, target = dd.get_test_logits(shape=(8, 1, 3, 3)), np.random.randint(size=(8, 1, 3, 3), low=0, high=2)
    >>> preds, target = torch.from_numpy(preds.astype(np.float32)), torch.from_numpy(target.astype(np.float32))
    >>> sc.forward(preds, target)
    tensor(6.8156)
    """

    def __init__(self, alpha, beta, loss_weights: Tuple):
        super().__init__()
        self.loss_weights = loss_weights
        self.focal_loss = FocalLoss(alpha=alpha, gamma=beta, logits=True)

    def forward(self, logits, target):
        # _, target = target_onehot.max(dim=1)
        bce = self.focal_loss(logits, target)
        # outputs = torch.sigmoid(logits)
        dice = dice_loss(logits, target)
        return bce * self.loss_weights[0] + dice * self.loss_weights[1]
