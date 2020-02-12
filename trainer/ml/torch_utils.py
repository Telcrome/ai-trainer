from abc import ABC
from enum import Enum
from functools import partial
from typing import Tuple, Union, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import optimizer
from torch.utils import data
from tqdm import tqdm

import trainer.lib as lib
import trainer.ml as ml

# If GPU is available, use GPU
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
IDENTIFIER = lib.create_identifier()


class ModelMode(Enum):
    """
    Used to differentiate what the model is currently doing.

    The following guidelines apply for the semantics of this enum:

    - Train asks for augmentation and other tricks during training (batch normalization, ...)
    - Eval does not require augmentation and is used for evaluation
    - Usage does not require ground truths
    """
    Train = 0
    Eval = 1
    Usage = 2


class TorchDataset(data.Dataset):
    """
    Wrapper around one dataset split to work with the torch.utils.data.Dataloader.
    This dataloader can be used to perform augmentations on multiple processes on the CPU and train on the GPU.
    """

    def __init__(self,
                 ds_path: str,
                 f: Union[Callable[[lib.Subject, ModelMode], Tuple[np.ndarray, np.ndarray]], partial],
                 split='',
                 mode: ModelMode = ModelMode.Train):
        super().__init__()
        self.ds = lib.Dataset.from_disk(ds_path)
        self.preprocessor = f
        self.split = split
        self.ss = self.ds.get_subject_name_list(split=self.split)
        self.mode = mode

    def get_torch_dataloader(self, batch_size=32, num_workers=1):
        return data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers)

    def __getitem__(self, item) -> Tuple[np.ndarray, np.ndarray]:
        """
        Uses the preprocessor that converts a subject to a training example.

        :param item: Name of a subject
        :return: Training example x, y
        """
        # print(f'item: {item}')
        s = self.ds.get_subject_by_name(self.ss[item])
        x, y = self.preprocessor(s, self.mode)
        # Cannot transformed to cuda tensors at this point,
        # because they do not seem to work in shared memory. Return numpy arrays instead.
        return x, y

    def __len__(self):
        return self.ds.get_subject_count(split=self.split)


ALL_TORCHSET_KEY = '_all_'


class TrainerModel(ABC):
    """
    TorchModel is a subclass of nn.Module with added functionality:

    - Name
    - Processing chain: Subject -> Augmented subject -> Input layer
    """

    def __init__(self,
                 model_name: str,
                 model: nn.Module,
                 opti: optimizer.Optimizer,
                 crit: Union[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], nn.Module]):
        super().__init__()
        self.model_name = model_name
        self.model, self.optimizer, self.criterion = model, opti, crit
        self.model = self.model.to(device)

    def print_summary(self):
        print(f"Capacity of the network: {get_capacity(self.model)}")

    def train_on_minibatch(self, training_example: Tuple[torch.Tensor, torch.Tensor]) -> float:
        x, y = training_example

        self.optimizer.zero_grad()
        y_ = self.model.forward(x)

        loss = self.criterion(y_, y.long())
        loss.backward()
        self.optimizer.step()

        batch_loss = loss.item()  # Loss, in the end, should be a single number
        return batch_loss

    def run_epoch(self, torch_loader: data.DataLoader, visboard: ml.VisBoard, epoch: int, batch_size: int, steps=-1):
        epoch_loss_sum = 0.

        steps = len(torch_loader) if steps == -1 else steps
        print(f'Starting epoch: {epoch} with {len(torch_loader) * batch_size} training examples and {steps} steps\n')
        loader_iterator = iter(torch_loader)
        with tqdm(total=steps, maxinterval=steps / 100) as pbar:
            for i in range(steps):
                x, y = next(loader_iterator)
                x, y = x.to(device), y.to(device)

                loss = self.train_on_minibatch((x, y))

                # Log metrics and loss
                epoch_loss_sum += (loss / batch_size)
                epoch_loss = epoch_loss_sum / (i + 1)
                visboard.add_scalar(f'loss/train epoch {epoch + 1}', epoch_loss, i)

                # Handle progress bar
                pbar.update()
                display_loss = epoch_loss_sum / (i + 1)
                pbar.set_description(f'Loss: {display_loss:05f}')
        print(f"\nEpoch result: {epoch_loss_sum / steps}\n")


def get_capacity(model: nn.Module) -> int:
    """
    Computes the number of parameters of a network.
    """
    import inspect

    # Instantiate, because model.parameters does not work on the class definition
    if inspect.isclass(model):
        model = model()

    return sum([p.numel() for p in model.parameters()])
