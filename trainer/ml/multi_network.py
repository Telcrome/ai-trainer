"""
Defines an encoder, a decoder head for semantic segmentation and a classification head.
"""
from typing import Tuple

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import trainer.lib as lib
import trainer.ml as ml
from trainer.ml import ModelMode


class SmallClassNet(nn.Module):
    def __init__(self):
        super(SmallClassNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


class MultiNetwork(ml.TrainerModel):

    def __init__(self):
        model = SmallClassNet()
        opti = optim.Adam(model.parameters(), lr=5e-3)
        crit = nn.CrossEntropyLoss()
        super().__init__(model_name='multi_model',
                         model=model,
                         opti=opti,
                         crit=crit)


if __name__ == '__main__':
    pass
