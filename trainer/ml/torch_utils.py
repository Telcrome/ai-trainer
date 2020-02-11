from abc import ABC, abstractmethod
from enum import Enum
from functools import partial
from typing import Tuple, Union, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import optimizer
from torch.utils import data
from tqdm import tqdm

import trainer.lib as lib

# If GPU is available, use GPU
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
IDENTIFIER = lib.create_identifier()


class ModelMode(Enum):
    """
    Used to differentiate what the model is currently doing.

    The following guidelines apply for the semantics of this enum:

    TODO: Explain the enum values.
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
                 crit: Union[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], nn.Module],
                 ds: lib.Dataset,
                 vis_board=None):
        super().__init__()
        self.model_name = model_name
        self.model, self.optimizer, self.criterion, self.ds = model, opti, crit, ds
        self.model = self.model.to(device)
        self._torch_sets = {
            ALL_TORCHSET_KEY: TorchDataset(self.ds.get_working_directory(), self.preprocess_segmap, split='')
        }
        self.vis_board = vis_board

    def print_summary(self):
        print(f"Capacity of the network: {get_capacity(self.model)}")

    def get_torch_dataset(self, split='', mode=ModelMode.Train):
        if not split:
            split = ALL_TORCHSET_KEY
        if split not in self._torch_sets:
            self._torch_sets[split] = TorchDataset(self.ds.get_working_directory(),
                                                   partial(self.preprocess_segmap, mode=mode),
                                                   # Python cant pickle lambda
                                                   split=split)
        return self._torch_sets[split]

    def train_on_minibatch(self, training_example: Tuple[torch.Tensor, torch.Tensor]) -> float:
        x, y = training_example

        self.optimizer.zero_grad()
        y_ = self.model(x)

        loss = self.criterion(y_, y)
        loss.backward()
        self.optimizer.step()

        batch_loss = loss.item()  # Loss, in the end, should be a single number
        return batch_loss

    def run_epoch(self, torch_loader: data.DataLoader, epoch: int, n: int, batch_size: int, steps=-1):
        print(f'Starting epoch: {epoch} with {n} training examples')
        epoch_loss_sum = 0.

        data_iter = iter(torch_loader)
        steps = n // batch_size if steps == -1 else steps
        for i in tqdm(range(steps)):
            x, y = data_iter.__next__()
            # x, y = seg_network.sample_minibatch(split='train')
            x, y = x.to(device), y.to(device)

            loss = self.train_on_minibatch((x, y))

            epoch_loss_sum += (loss / batch_size)
            epoch_loss = epoch_loss_sum / (i + 1)
            self.vis_board.add_scalar(f'loss/train epoch {epoch + 1}', epoch_loss, i)
        print(f"Epoch result: {epoch_loss_sum / n}")

    def train_supervised(self,
                         structure_template: str,
                         train_split='',
                         max_epochs=50,
                         batch_size=4,
                         load_latest_state=True,
                         num_workers=2):

        train_loader = data.DataLoader(
            self.get_torch_dataset(split=train_split, mode=ModelMode.Train),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers)

        N = self.ds.get_subject_count(split=train_split)

        if load_latest_state:
            self.load_from_dataset(structure_template=structure_template)

        for epoch in range(max_epochs):
            # with torch.no_grad():
            #     # Visualize model output
            #     for fig in self.visualize_prediction(machine_loader):
            #         fig.suptitle("B8 Stuff")
            #         self.visboard.add_figure(fig, group_name=f'Before Epoch{epoch}, B8')
            #     for fig in self.visualize_prediction(test_loader):
            #         fig.suptitle("Test Set")
            #         self.visboard.add_figure(fig, group_name=f'Before Epoch{epoch}, Test')
            # for fig in seg_network.visualize_prediction(train_loader):
            #     fig.suptitle("Train Set")
            #     visboard.add_figure(fig, group_name=f'Before Epoch{epoch}, Train')
            self.run_epoch(train_loader, epoch, N, batch_size=batch_size)

            # Save model weights
            self.save_to_dataset('bone', epoch)

    def save_to_dataset(self):
        save_obj = {
            'model': self.model.state_dict(),
            'optim': self.optimizer.state_dict()
        }
        self.ds.save_model_state(self.model_name, save_obj)

    def load_from_dataset(self, structure_template: str, epoch=-1):
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def preprocess_segmap(s: lib.Subject, mode: ModelMode = ModelMode.Train) -> Tuple[np.ndarray, np.ndarray]:
        """
        Provides the preprocessing chain to extract a training example from a subject.

        :param s: One subject
        :param mode: ModelMode for the preprocessor. Train is used for training, eval is used for testing, Usage for use
        :return: The training example (x, y), of type torch.Tensor
        """
        pass

    @staticmethod
    @abstractmethod
    def preprocess_class(s: lib.Subject, mode: ModelMode = ModelMode.Train) -> Tuple[np.ndarray, np.ndarray]:
        """

        :param s: One subject
        :param mode: ModelMode for the preprocessor. Train is used for training, eval is used for testing, Usage for use
        :return:
        """
        pass

    @abstractmethod
    def visualize_input_batch(self, te: Tuple[np.ndarray, np.ndarray]) -> plt.Figure:
        """
        Needs to be implemented by the subclass, because different networks.

        :return: A matplotlib.figure
        """
        pass


def get_capacity(model: nn.Module) -> int:
    """
    Computes the number of parameters of a network.
    """
    import inspect

    # Instantiate, because model.parameters does not work on the class definition
    if inspect.isclass(model):
        model = model()

    return sum([p.numel() for p in model.parameters()])
