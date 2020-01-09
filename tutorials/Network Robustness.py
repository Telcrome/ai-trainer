#!/usr/bin/env python
# coding: utf-8

# Adapted from https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

from trainer.bib import create_identifier

IDENTIFIER = create_identifier()


def normalize_mnist(x):
    return x * 2 - 1


# MNIST Test dataset and dataloader declaration
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize_mnist
    ])),
    batch_size=32, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize_mnist
    ])),
    batch_size=1, shuffle=True)


def visualize_one_input():
    _, (images, labels) = next(enumerate(train_loader, 0))
    text_index = 0
    im, label = images[text_index, 0, :, :].numpy(), labels[text_index].item()
    plt.title(f"True Label: {label}")
    sns.heatmap(im)
    plt.savefig("MNIST_Example.png")


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()


# create grid of images


class NamedModel(nn.Module):
    def __init__(self, model_name: str):
        super(NamedModel, self).__init__()
        self.name = model_name


class ConvNet(NamedModel):
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


class FCNet(NamedModel):
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


device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

# Initialize the network
models = [FCNet().to(device), ConvNet().to(device)]


def train_model(model: NamedModel, writer: SummaryWriter):
    img_grid = torchvision.utils.make_grid(images)
    writer.add_image(f'Input for {model.name}', img_grid)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    EPOCHS = 5

    for epoch in range(EPOCHS):

        running_loss, print_loss = 0.0, "-1."

        for batch_id, data in enumerate(train_loader, 0):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if batch_id > 0:
                writer.add_scalar(f'Loss/train{epoch}', running_loss / batch_id, batch_id)


def fgsm_attack(image: torch.Tensor, epsilon: float, data_grad: torch.Tensor):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def test(model: NamedModel, device: torch.device, test_loader: torch.utils.data.DataLoader, epsilon: float, writer):
    wrong, correct, adv_success = 0, 0, 0
    adv_examples = []

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        data.requires_grad = True  # This enables a gradient based attack such as fgsm

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]

        if init_pred.item() != target.item():
            wrong += 1
        else:
            # Prediction was correct, try to fool the network
            correct += 1

            loss = F.nll_loss(output, target)

            model.zero_grad()

            loss.backward()

            data_grad = data.grad.data

            perturbed_data = fgsm_attack(data, epsilon, data_grad)

            output = model(perturbed_data)

            adversarial_pred = output.max(1, keepdim=True)[1]  # max returns a tuple (values, indices)
            if np.random.rand() > 0.9 or True:
                perturbed_arr = perturbed_data.detach().cpu().numpy()
                writer.add_image(f'Adversarial Example with Epsilon={epsilon}', perturbed_arr[0])
                adv_examples.append((target.item(), adversarial_pred.item(), perturbed_arr))

            if adversarial_pred.item() != target.item():
                adv_success += 1

    print(f'Correct: {correct}; Wrong: {wrong}')
    print(f'Adversarial Successes: {adv_success}')
    acc = correct / (wrong + correct)
    return acc, adv_examples


def run_adv_attacks(model):
    epsilons = [0, .05, .1, .15, .2, .25, .3]
    for epsilon in epsilons:
        acc, meta = test(model, device, test_loader, epsilon=epsilon)
        true_target, adv_pred, adv_ex = random.choice(meta)
        plt.title(f"Epsilon: {epsilon}, True Target: {true_target}, Pred: {adv_pred}")
        plt.imshow(adv_ex.squeeze())
        plt.show()
        print(f"Accuracy for epsilon={epsilon}: {acc}")


def visualize_model_weights():
    W = model.state_dict()

    for key in W.keys():
        print(key)

    arr = W['conv2.weight'].cpu().numpy()
    print(arr.shape)
    return W


def vis_conv_weights(arr: np.ndarray):
    plt.title(f"Convolutional Weights")

    fig_width = 12.  # inches
    fig_height = 12.

    cmaps = ['Greens', 'plasma']

    f, axarr = plt.subplots(arr.shape[0], arr.shape[1], figsize=(fig_width, fig_height))
    for depth in range(arr.shape[0]):
        for channel in range(arr.shape[1]):
            plot_index = 1 + depth * arr.shape[1] + channel
            # print(f"ASDF: {plot_index}")
            im = arr[depth, channel, :, :]
            axarr[depth, channel].imshow(im, cmap=cmaps[depth % 2])
            axarr[depth, channel].axis('off')

            # plt.subplot(arr.shape[0], arr.shape[1], plot_index)
            # plt.imshow(arr[depth, channel, : , :])
    plt.savefig("weights.png")
    plt.show()


if __name__ == '__main__':
    model = models[0]
    writer = SummaryWriter(f'runs/{model.name}_{IDENTIFIER}')
    writer.add_graph(model, images.to(device))

    train_model(model, writer)
