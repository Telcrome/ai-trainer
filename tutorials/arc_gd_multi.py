"""
Alternates between optimizing feature & action network
and task networks.
"""
from typing import Tuple, List
import random

import numpy as np
from numba import njit
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import trainer.lib as lib
import trainer.ml as ml
from trainer.demo_data.arc import extract_train_test


def single_preprocessor(s: lib.Subject, mode: ml.ModelMode) -> List[Tuple[List[np.ndarray], List[np.ndarray], str]]:
    train_examples, test_examples = extract_train_test(s)

    def extract_pair(im: lib.ImStack) -> Tuple[np.ndarray, np.ndarray]:
        # Read from imagestack
        x, y = im.values()[0, :, :, :], im.semseg_masks[0].values()[:, :, :]
        x, y = np.rollaxis(x, 2, 0).astype(np.float32), np.rollaxis(y, 2, 0).astype(np.float32)
        x, y = ml.one_hot_to_cont(x) + 1, ml.one_hot_to_cont(y) + 1

        # Pad to 30x30
        x, y = ml.pad(x), ml.pad(y)

        return ml.cont_to_ont_hot(x, n_values=11), y

    train_pairs = [extract_pair(im) for im in train_examples]
    test_pairs = [extract_pair(im) for im in test_examples]
    return train_pairs, test_pairs, s.name


class TaskNetwork(nn.Module):

    def __init__(self, hidden_depth=20):
        super(TaskNetwork, self).__init__()
        self.first_cell = nn.Conv2d(
            in_channels=hidden_depth,
            out_channels=hidden_depth,
            kernel_size=1
        )
        self.middle_cell = nn.Conv2d(
            in_channels=hidden_depth,
            out_channels=hidden_depth,
            kernel_size=1
        )
        self.output_cell = nn.Conv2d(
            in_channels=hidden_depth,
            out_channels=hidden_depth,
            kernel_size=1
        )

    def forward(self, x):
        x = F.relu(self.first_cell(x))
        x = F.relu(self.middle_cell(x))
        x = self.output_cell(x)
        return x


class FeatureNet(nn.Module):

    def __init__(self, network_depth=5, in_channels=11, hidden_depth=20):
        super(FeatureNet, self).__init__()
        self.start_cell = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_depth,
            kernel_size=3,
            padding=1
        )
        self.cells = nn.ModuleList(
            [nn.Conv2d(
                in_channels=hidden_depth,
                out_channels=hidden_depth,
                kernel_size=3,
                padding=1) for _ in range(network_depth)]
        )

    def forward(self, x):
        x = F.relu(self.start_cell(x))
        for cell in self.cells:
            x = F.relu(cell(x))
        return x


class ActionNet(nn.Module):

    def __init__(self, network_depth=5, out_channels=11, hidden_depth=20):
        super(ActionNet, self).__init__()
        self.cells = nn.ModuleList(
            [nn.Conv2d(
                in_channels=hidden_depth,
                out_channels=hidden_depth,
                kernel_size=3,
                padding=1
            ) for _ in range(network_depth)]
        )
        self.end_cell = nn.Conv2d(
            in_channels=hidden_depth,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):
        for cell in self.cells:
            x = F.relu(cell(x))
        x = F.relu(self.end_cell(x))
        return x


class MultiTaskNetwork(nn.Module):

    def __init__(self, hidden_depth=200, feature_depth=0, action_depth=0):
        super(MultiTaskNetwork, self).__init__()
        self.hidden_depth = hidden_depth
        self.feature_network = FeatureNet(in_channels=11, hidden_depth=hidden_depth, network_depth=feature_depth)
        self.task_networks: nn.ModuleDict = nn.ModuleDict({})
        self.action_network = ActionNet(out_channels=11, hidden_depth=hidden_depth, network_depth=action_depth)
        self.activation_layer = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor, task_name: str, opti):
        """
        If a forward call is made for an unknown task, a new task network is created
        """
        if task_name not in self.task_networks:
            self.task_networks[task_name] = TaskNetwork(hidden_depth=self.hidden_depth)
            opti.add_param_group({'params': self.task_networks[task_name].parameters()})

        x = self.feature_network(x)
        x = self.task_networks[task_name](x)
        x = self.action_network(x)
        # return self.activation_layer(x)
        return x


if __name__ == '__main__':
    BATCH_SIZE = 1
    EPOCHS = 1000
    EACH_TASK_EPOCHS = 5

    sess = lib.Session()
    # Prepare data
    train_set = ml.InMemoryDataset('arc', 'training', single_preprocessor, mode=ml.ModelMode.Train)
    test_set = ml.InMemoryDataset('arc', 'evaluation', single_preprocessor, mode=ml.ModelMode.Eval)

    net = MultiTaskNetwork()  # .to(ml.torch_device)
    # x = train_set.get_random_batch()
    #
    #
    # with torch.no_grad():
    #     inp = torch.from_numpy(np.ones((32, 10, 5, 5), dtype=np.float32))
    #     some_result = task_network(inp)

    # train_loader = train_set.get_torch_dataloader(batch_size=BATCH_SIZE)

    # criterion = nn.CrossEntropyLoss()
    criterion = ml.FocalLoss(alpha=1., gamma=2., logits=True, reduce=True)
    optimizer = optim.Adam(
        net.parameters(),
        lr=5e-2
    )
    # loop over all training tasks in each epoch
    for epoch in range(EPOCHS):

        for _ in range(10):
            task = random.choice(train_set)
            train_pairs, test_pairs, s_name = task

            situation, target = random.choice(train_pairs)
            situation = torch.from_numpy(situation).float().unsqueeze(0)
            target = torch.from_numpy(target).long().unsqueeze(0)

            with torch.no_grad():
                prediction = net(situation, s_name, optimizer)

            pred_img = ml.one_hot_to_cont(prediction.squeeze().numpy())
            lib.logger.debug_var(pred_img)

        running_loss = 0.

        # Looping over all training tasks
        p_bar = tqdm(enumerate(train_set, 0), total=len(train_set))
        for i, task in p_bar:
            train_pairs, test_pairs, s_name = task
            # Try to improve on the task specific network
            # TODO: Do not iterate through all training examples, this favors tasks with bigger datasets

            for situation, target in train_pairs:
                situation = torch.from_numpy(situation).float().unsqueeze(0)
                target = torch.from_numpy(target).long().unsqueeze(0)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(situation, s_name, optimizer)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

                running_loss = running_loss + (loss.item() / len(train_pairs))
            if i > 0:
                p_bar.set_description(f'Loss: {running_loss / i}')
                p_bar.set_postfix_str(f'Optimizing task {s_name}')

            # # Temporary check to visualize the last training example
            # pred_img = ml.one_hot_to_cont(outputs.detach().squeeze().numpy())
            # lib.logger.debug_var(pred_img)
            #
            # # Logging and statistics of the performance of the network corresponding to task
            # # TODO
            # for situation, target in test_pairs:
            #     situation = torch.from_numpy(situation).float().unsqueeze(0)
            #     target = torch.from_numpy(target).long().unsqueeze(0)
            #
            #     with torch.no_grad():
            #         prediction = net(situation, s_name, optimizer)
            #
            #     pred_img = ml.one_hot_to_cont(prediction.squeeze().numpy())
            #     lib.logger.debug_var(pred_img)
            # print(prediction)

    print('Finished Training')
