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

        # Channels last to channels first
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

    def __init__(self, network_depth=1, in_channels=11, hidden_depth=20):
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

    def __init__(self, network_depth=1, out_channels=11, hidden_depth=20):
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

    def __init__(self, hidden_depth=10, feature_depth=3, action_depth=3):
        super(MultiTaskNetwork, self).__init__()
        self.hidden_depth = hidden_depth
        self.feature_network = FeatureNet(in_channels=11, hidden_depth=hidden_depth, network_depth=feature_depth)
        # self.task_networks: nn.ModuleDict = nn.ModuleDict({})
        self.task_network: nn.Module = TaskNetwork(hidden_depth=hidden_depth)
        self.action_network = ActionNet(out_channels=11, hidden_depth=hidden_depth, network_depth=action_depth)
        self.activation_layer = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor, task_name: str, opti):
        """
        If a forward call is made for an unknown task, a new task network is created
        """
        # if task_name not in self.task_networks:
        #     self.task_networks[task_name] = TaskNetwork(hidden_depth=self.hidden_depth).to(ml.torch_device)
        #     opti.add_param_group({'params': self.task_networks[task_name].parameters()})

        x = self.feature_network(x)
        # x = self.task_networks[task_name](x)
        x = self.task_network(x)
        x = self.action_network(x)
        # x = self.activation_layer(x)
        x = F.log_softmax(x, dim=1)
        return x


if __name__ == '__main__':
    BATCH_SIZE = 1
    EPOCHS = 50
    STEPS_PER_TASK = 5

    sess = lib.Session()
    # Prepare data
    train_set = ml.InMemoryDataset('arc', 'training', single_preprocessor, mode=ml.ModelMode.Train)
    test_set = ml.InMemoryDataset('arc', 'evaluation', single_preprocessor, mode=ml.ModelMode.Eval)

    net = MultiTaskNetwork().to(ml.torch_device)
    # x = train_set.get_random_batch()
    #
    #
    # with torch.no_grad():
    #     inp = torch.from_numpy(np.ones((32, 10, 5, 5), dtype=np.float32))
    #     some_result = task_network(inp)

    # train_loader = train_set.get_torch_dataloader(batch_size=BATCH_SIZE)

    criterion = nn.CrossEntropyLoss()
    # criterion = ml.FocalLoss(alpha=1., gamma=2., logits=False, reduce=True)
    optimizer = optim.Adam(
        net.parameters(),
        lr=5e-3
    )
    # loop over all training tasks in each epoch
    for epoch in range(EPOCHS):

        viss = []
        for _ in range(2):
            # task = random.choice(train_set)
            task0 = train_set[1]
            train_pairs, test_pairs, s_name = task0

            train_pred = random.choice([True, False])
            situation, target = random.choice(train_pairs) if train_pred else random.choice(test_pairs)
            situation = torch.from_numpy(situation).float().unsqueeze(0).to(ml.torch_device)
            target = torch.from_numpy(target).long().unsqueeze(0)

            with torch.no_grad():
                prediction = net(situation, s_name, optimizer)

            pred_img = ml.one_hot_to_cont(prediction.squeeze().cpu().numpy())
            viss.append(
                (pred_img, f'Pred: {s_name}, Train: {train_pred}')
            )
            viss.append(
                (target[0].numpy(), f'Target: {s_name}, Train: {train_pred}')
            )
        lib.logger.debug_var(viss)

        running_loss = 0.

        # Looping over all training tasks
        p_bar = tqdm(enumerate(train_set, 0), total=len(train_set))
        for i, task in p_bar:
            task0 = train_set[1]
            train_pairs, test_pairs, s_name = task0
            # Try to improve on the task specific network
            # TODO: Do not iterate through all training examples, this favors tasks with bigger datasets

            loss = torch.zeros(1, requires_grad=True).to(ml.torch_device)
            optimizer.zero_grad()

            # Task specific training, accumulate gradients
            for situation, target in train_pairs:
                situation = torch.from_numpy(situation).float().unsqueeze(0).to(ml.torch_device)
                target = torch.from_numpy(target).long().unsqueeze(0).to(ml.torch_device)
                outputs = net(situation, s_name, optimizer)
                loss += criterion(outputs, target)

            running_loss = running_loss + (loss.item() / len(train_pairs))
            loss.backward()
            optimizer.step()

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
