import os
from enum import Enum
from typing import Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

from trainer.bib import create_identifier


class TestSet(Enum):
    MNIST = "MNIST"


def load_testset(dataset: TestSet, local_path='./data'):
    if dataset == TestSet.MNIST:
        def normalize_mnist(x):
            return x * 2 - 1

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(local_path, train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize_mnist
            ])),
            batch_size=32, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(local_path, train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize_mnist
            ])),
            batch_size=1, shuffle=True)

        return train_loader, test_loader


# If GPU is available, use GPU
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
IDENTIFIER = create_identifier()


class TorchModel(nn.Module):
    """
    Torchmodels are a subclass of nn.Module with added functionality:
    - Name
    """

    def __init__(self, model_name: str):
        super(TorchModel, self).__init__()
        self.name = model_name


def instantiate_model(model_definition: TorchModel, weights_path='', data_loader=None) -> Tuple[
    TorchModel, SummaryWriter]:
    model = model_definition().to(device)
    writer = SummaryWriter(f'tb/{model.name}_{IDENTIFIER}')
    if data_loader is not None:
        test_input = iter(data_loader).__next__()[0].to(device)
        writer.add_graph(model, test_input)

    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path))

    return model, writer


def fgsm_attack(image: torch.Tensor, epsilon: float, data_grad: torch.Tensor):
    """
    Fast gradient sign method attack as proposed by:
    https://arxiv.org/pdf/1712.07107.pdf
    """
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def visualize_input_batch(data_loader, writer: SummaryWriter, name="Input Example"):
    dataiter = iter(data_loader)
    images, labels = dataiter.next()
    img_grid = torchvision.utils.make_grid(images)
    writer.add_image(name, img_grid)


def train_model(train_loader, model: TorchModel, writer: SummaryWriter, save_path='', verbose=True):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    N = len(train_loader)
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
                print_loss = running_loss / batch_id
                writer.add_scalar(f'Loss/training epoch {epoch}', print_loss, batch_id)
        if verbose:
            print(f"Epoch: {epoch}; Loss: {print_loss}")
    return model.state_dict()


def visualize_model_weights(model: TorchModel, writer: SummaryWriter):
    for i, layer in enumerate(model.children()):
        if isinstance(layer, nn.Linear):
            # Visualize a fully connected layer
            pass
        elif isinstance(layer, nn.Conv2d):
            # Visualize a convolutional layer
            W = layer.weight
            b = layer.bias
            for d in range(W.shape[0]):
                image_list = np.array([W[d, c, :, :].detach().cpu().numpy() for c in range(W.shape[1])])
                placeholder_arr = torch.from_numpy(np.expand_dims(image_list, 1))
                img_grid = torchvision.utils.make_grid(placeholder_arr, pad_value=1)
                writer.add_image(f"layer{i}", img_grid)
