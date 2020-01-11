import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from trainer.ml.torch_utils import instantiate_model, TorchModel, fgsm_attack, train_model, load_testset, TestSet
from trainer.ml.torch_utils import visualize_input_batch, visualize_model_weights, perform_adversarial_testing


class FCConvNet(TorchModel):
    def __init__(self):
        super(FCConvNet, self).__init__(model_name="fcconvnet")
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 10, kernel_size=1)
        self.global_pool = nn.AvgPool2d(4)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3(x))
        x = self.global_pool(x).view(-1, 10)
        res = F.log_softmax(x, dim=1)
        return res


class ConvNet(TorchModel):
    def __init__(self):
        super(ConvNet, self).__init__(model_name="convnet")
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class FCNet(TorchModel):
    def __init__(self):
        super(FCNet, self).__init__(model_name='fcnet')
        self.fc1 = nn.Linear(28 * 28, 28 * 14)
        self.fc2 = nn.Linear(28 * 14, 28 * 8)
        self.output = nn.Linear(28 * 8, 10)

    def forward(self, x):
        x = x.view(-1, self.fc1.in_features)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return F.log_softmax(x, dim=1)


def full_analysis(model_definition: TorchModel, weights_path):
    train_loader, test_loader = load_testset(TestSet.MNIST)

    net, writer = instantiate_model(model_definition, weights_path=weights_path, data_loader=train_loader)

    visualize_input_batch(train_loader, writer)

    if not os.path.exists(weights_path):
        fc_state = train_model(train_loader, net, writer, test_loader=test_loader)
        torch.save(fc_state, weights_path)

    visualize_model_weights(net, writer)

    train_loader, test_loader = load_testset(TestSet.MNIST, batch_size=1)

    epsilons = [.1, .2, .3]
    for epsilon in epsilons:
        perform_adversarial_testing(net, test_loader, epsilon, writer)


if __name__ == '__main__':
    full_analysis(FCNet, 'fc_net.pt')
    full_analysis(ConvNet, 'conv_net.pt')
    full_analysis(FCConvNet, 'fcconv_net.pt')

    from trainer.ml.torch_utils import compare_architectures
    from trainer.ml.visualization import VisBoard
    num_parameters = compare_architectures([FCNet, ConvNet, FCConvNet], VisBoard(run_name='architecture_comparison'))
    print(num_parameters)
    # print(f"Parameters of {model_definition.name}: {num_parameters}")
