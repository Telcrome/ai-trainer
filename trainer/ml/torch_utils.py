import os
import random
from enum import Enum
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

from trainer.bib import create_identifier, get_img_from_fig
from trainer.ml.visualization import VisBoard


class TestSet(Enum):
    MNIST = "MNIST"


def load_testset(dataset: TestSet, local_path='./data', batch_size=32):
    if dataset == TestSet.MNIST:
        def normalize_mnist(x):
            return x * 2 - 1

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(local_path, train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize_mnist
            ])),
            batch_size=batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(local_path, train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize_mnist
            ])),
            batch_size=batch_size, shuffle=True)

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
    TorchModel, VisBoard]:
    model = model_definition().to(device)
    visboard = VisBoard(run_name=f'{model.name}_{IDENTIFIER}')
    if data_loader is not None:
        test_input = iter(data_loader).__next__()[0].to(device)
        visboard.writer.add_graph(model, test_input)

    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path))

    return model, visboard


def oneshot_attack(image: torch.Tensor, max_change_percentage: float, data_grad: torch.Tensor, data_range=(-1., 1.)):
    """
    Oneshot attack, takes one step against the gradient direction to increase the error.
    """
    # sign_data_grad = data_grad.sign()
    epsilon = (((max_change_percentage * image) / data_grad).sum() / image.numel()).item()
    perturbed_image = image + epsilon * data_grad
    perturbed_image = torch.clamp(perturbed_image, data_range[0], data_range[1])
    return perturbed_image


def fgsm_attack(image: torch.Tensor, epsilon: float, data_grad: torch.Tensor, data_range=(-1., 1.)):
    """
    Fast gradient sign method attack as proposed by:
    https://arxiv.org/pdf/1712.07107.pdf

    FGSM is an attack for an infinity-norm bounded adversary.
    """
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, data_range[0], data_range[1])
    return perturbed_image


def visualize_input_batch(data_loader, visboard: VisBoard, name="Input Example"):
    dataiter = iter(data_loader)
    images, labels = dataiter.next()
    img_grid = torchvision.utils.make_grid(images)
    visboard.writer.add_image(name, img_grid)
    for batch_id in range(images.shape[0]):
        fig, ax1 = plt.subplots()

        ax1.set_title(f"Input Example for {labels[batch_id]}")
        sns.heatmap(images[batch_id, 0, :, :])
        visboard.add_figure(fig, group_name=name)


def train_model(train_loader, model: TorchModel, visboard: VisBoard, verbose=True, test_loader=None,
                EPOCHS=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    N = len(train_loader)

    last_epoch_loss, running_loss = -1, 0.  # Used for early stopping

    for epoch in range(EPOCHS):

        if 0. <= last_epoch_loss < running_loss:
            break

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
                visboard.writer.add_scalar(f'Loss/training epoch {epoch + 1}', print_loss, batch_id)
        if verbose:
            print(f"Epoch: {epoch + 1}; Loss: {print_loss}")
            last_epoch_loss = running_loss
            if test_loader is not None:
                test_acc = measure_classification_accuracy(model, test_loader)
                print(f"Accuracy: {test_acc}")
                visboard.writer.add_scalar(f'Accuracy/test', test_acc, epoch + 1)
    return model.state_dict()


def measure_classification_accuracy(net: nn.Module, data_loader: torch.utils.data.DataLoader):
    with torch.no_grad():
        all, correct = 0, 0
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            pred = output.max(1, keepdim=True)[1].squeeze()
            correct_tmp = (target == pred).sum()
            correct += int(correct_tmp.cpu().numpy())
            all += pred.shape[0]
        acc = correct / all
    return acc


def visualize_model_weights(model: TorchModel, visboard: VisBoard):
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
                visboard.writer.add_image(f"{model.name}_layer_{i}", img_grid)


def perform_adversarial_testing(model: TorchModel, test_loader: torch.utils.data.DataLoader, epsilon: float,
                                visboard: VisBoard, test_number=100):
    wrong, correct, adv_success = 0, 0, 0
    adv_examples = []

    for data, target in test_loader:
        if wrong + correct >= test_number:
            break
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
                original_arr = data.detach().cpu().numpy()
                perturbed_arr = perturbed_data.detach().cpu().numpy()

                fig, (ax1, ax2) = plt.subplots(1, 2)

                ax1.set_title(f"Original: {target.item()}, Pred: {init_pred.item()}")
                sns.heatmap(original_arr[0].squeeze(), ax=ax1)

                ax2.set_title(f"Epsilon: {epsilon}, True Target: {target.item()}, Pred: {adversarial_pred.item()}")
                sns.heatmap(perturbed_arr[0].squeeze(), ax=ax2)

                visboard.add_figure(fig, group_name=f"Adversarial Example, Epsilon={epsilon}")

                adv_examples.append((target.item(), adversarial_pred.item(), perturbed_arr))

            if adversarial_pred.item() != target.item():
                adv_success += 1

    print(f'Correct: {correct}; Wrong: {wrong}')
    print(f'Adversarial Successes: {adv_success}')
    acc = correct / (wrong + correct)
    return acc, adv_examples


def compare_architectures(models: List[nn.Module], writer: VisBoard) -> List[int]:
    import inspect

    # Instantiate, because model.parameters does not work on the class definition
    instantiated_list = []
    for model in models:
        if inspect.isclass(model):
            instantiated_list.append(model())
        else:
            instantiated_list.append(model)
    models = instantiated_list

    params = [sum([p.numel() for p in model.parameters()]) for model in models]

    fig, ax1 = plt.subplots()

    ax1.set_title(f"Number of Parameters")
    sns.barplot(x=[m.name for m in models], y=params, ax=ax1)

    writer.add_figure(fig, group_name='Parameter Number')
    return params
