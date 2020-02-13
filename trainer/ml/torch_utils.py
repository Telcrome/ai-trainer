from abc import ABC, abstractmethod
from enum import Enum
from functools import partial
from typing import Tuple, Union, Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
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


class TrainerMetric(ABC):
    """
    Base class of trainer metrics
    """

    def evaluate(self, preds: np.ndarray, targets: np.ndarray):
        assert (preds.shape[0] == targets.shape[0]), f'Batch sizes do not match: {preds.shape} and {targets.shape}'
        batch_size = preds.shape[0]
        for batch_id in range(batch_size):
            self.update(preds[batch_id], targets[batch_id])

    @abstractmethod
    def update(self, prediction: np.ndarray, target: np.ndarray):
        pass

    @abstractmethod
    def get_result(self):
        pass


class AccuracyMetric(TrainerMetric):

    def __init__(self):
        self.preds = []
        self.targets = []

    def update(self, prediction: np.ndarray, target: np.ndarray):
        if len(prediction.shape) != len(target.shape):
            # the prediction seems to be given in logits or class probabilities
            prediction = np.argmax(prediction)
            if type(prediction) != np.ndarray:
                prediction = np.array(prediction)

        self.preds.append(prediction)
        self.targets.append(target)

    def get_result(self):
        return accuracy_score(self.targets, self.preds)


class TrainerModel(ABC):
    """
    TorchModel is a subclass of nn.Module with added functionality:

    - Name
    - Processing chain: Subject -> Augmented subject -> Input layer
    - Visboard

    Implements standard training and evaluation methods.
    """

    def __init__(self,
                 model_name: str,
                 model: nn.Module,
                 opti: optimizer.Optimizer,
                 visboard: ml.VisBoard,
                 crit: Union[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], nn.Module]):
        super().__init__()
        self.model_name = model_name
        self.model, self.optimizer, self.criterion = model, opti, crit
        self.model = self.model.to(device)
        self.visboard = visboard

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

    def run_epoch(self, torch_loader: data.DataLoader, epoch: int, batch_size: int, steps=-1):
        self.model.train()
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
                self.visboard.add_scalar(f'loss/train epoch {epoch + 1}', epoch_loss, i)

                # Handle progress bar
                pbar.update()
                display_loss = epoch_loss_sum / (i + 1)
                pbar.set_description(f'Loss: {display_loss:05f}')
        print(f"\nEpoch result: {epoch_loss_sum / steps}\n")

    def evaluate(self, eval_set: TorchDataset, evaluator: TrainerMetric, batch_size=1, num_workers=2):
        self.model.eval()
        steps = len(eval_set)
        eval_loader = iter(eval_set.get_torch_dataloader(batch_size=batch_size, num_workers=num_workers))
        with torch.no_grad():  # Testing does not require gradients
            with tqdm(total=steps, maxinterval=steps / 100) as pbar:
                for i in range(steps):
                    x, y = next(eval_loader)
                    x = x.to(device)
                    y_ = self.model.forward(x).cpu()

                    y, y_ = y.numpy(), y_.numpy()

                    evaluator.update(y_, y)

                    pbar.update()
                    pbar.set_description(f"{evaluator}")


def get_capacity(model: nn.Module) -> int:
    """
    Computes the number of parameters of a network.
    """
    import inspect

    # Instantiate, because model.parameters does not work on the class definition
    if inspect.isclass(model):
        model = model()

    return sum([p.numel() for p in model.parameters()])
