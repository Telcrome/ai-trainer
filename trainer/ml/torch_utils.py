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
                 f: Union[Callable[[lib.Subject], Tuple[np.ndarray, np.ndarray]], partial],
                 split=''):
        super().__init__()
        self.ds = lib.Dataset.from_disk(ds_path)
        self.preprocessor = f
        self.split = split
        self.ss = self.ds.get_subject_name_list(split=self.split)

    def __getitem__(self, item):
        # print(f'item: {item}')
        s = self.ds.get_subject_by_name(self.ss[item])
        x, y = self.preprocessor(s)
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
                 batch_size=4,
                 vis_board=None):
        super().__init__()
        self.name = model_name
        self.model, self.optimizer, self.criterion, self.ds, self.batch_size = model, opti, crit, ds, batch_size
        self.model = self.model.to(device)
        self._torch_sets = {
            ALL_TORCHSET_KEY: TorchDataset(self.ds.get_working_directory(), self.preprocess, split='')
        }
        self.vis_board = vis_board

    def get_torch_dataset(self, split='', mode=ModelMode.Train):
        if not split:
            split = ALL_TORCHSET_KEY
        if split not in self._torch_sets:
            self._torch_sets[split] = TorchDataset(self.ds.get_working_directory(),
                                                   partial(self.preprocess, mode=mode),  # Python cant pickle lambda
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

    def run_epoch(self, torch_loader: data.DataLoader, epoch: int, n: int, batch_size: int):
        print(f'Starting epoch: {epoch} with {n} training examples')
        epoch_loss_sum = 0.

        data_iter = iter(torch_loader)
        for i in tqdm(range(n // batch_size)):
            x, y = data_iter.__next__()
            # x, y = seg_network.sample_minibatch(split='train')
            x, y = x.to(device), y.to(device)

            loss = self.train_on_minibatch((x, y))

            epoch_loss_sum += (loss / batch_size)
            epoch_loss = epoch_loss_sum / (i + 1)
            self.vis_board.add_scalar(f'loss/train epoch {epoch + 1}', epoch_loss, i)
        print(f"Epoch result: {epoch_loss_sum / n}")

    def save_to_dataset(self, structure_template: str, epoch: int):
        self.ds.add_binary(
            f'model_{self.name}_{structure_template}_{epoch}',
            self.model.state_dict(),
            lib.BinaryType.TorchStateDict.value,
        )
        # TODO: Save metrics using binary meta info

    def load_from_dataset(self):
        pass

    @staticmethod
    @abstractmethod
    def preprocess(s: lib.Subject, mode: ModelMode = ModelMode.Train) -> Tuple[np.ndarray, np.ndarray]:
        """
        Provides the preprocessing chain to extract a training example from a subject.

        :param s: One subject
        :param mode: ModelMode for the preprocessor. Train is used for training, eval is used for testing, Usage for use
        :return: The training example (x, y), of type torch.Tensor
        """
        pass

    @abstractmethod
    def visualize_input_batch(self, te: Tuple[np.ndarray, np.ndarray]) -> plt.Figure:
        """
        Needs to be implemented by the subclass, because different networks.

        :return: A matplotlib.figure
        """
        pass


# def instantiate_model(model_definition: TorchModel, weights_path='', data_loader=None) -> Tuple[TorchModel, VisBoard]:
#     model = model_definition().to(device)
#     visboard = VisBoard(run_name=f'{model.name}_{IDENTIFIER}')
#     if data_loader is not None:
#         test_input = iter(data_loader).__next__()[0].to(device)
#         visboard.writer.add_graph(model, test_input)
#
#     if weights_path and os.path.exists(weights_path):
#         model.load_state_dict(torch.load(weights_path))
#
#     return model, visboard


# def visualize_model_weights(model: TorchModel, visboard: VisBoard):
#     for i, layer in enumerate(model.children()):
#         if isinstance(layer, nn.Linear):
#             # Visualize a fully connected layer
#             pass
#         elif isinstance(layer, nn.Conv2d):
#             # Visualize a convolutional layer
#             W = layer.weight
#             b = layer.bias
#             for d in range(W.shape[0]):
#                 image_list = np.array([W[d, c, :, :].detach().cpu().numpy() for c in range(W.shape[1])])
#                 placeholder_arr = torch.from_numpy(np.expand_dims(image_list, 1))
#                 img_grid = torchvision.utils.make_grid(placeholder_arr, pad_value=1)
#                 visboard.writer.add_image(f"{model.name}_layer_{i}", img_grid)
#
#
def get_capacity(model: nn.Module) -> int:
    """
    Computes the number of parameters of a network.
    """
    import inspect

    # Instantiate, because model.parameters does not work on the class definition
    if inspect.isclass(model):
        model = model()

    return sum([p.numel() for p in model.parameters()])
