import os
from enum import Enum
from typing import Tuple, List, Union, Callable
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import optimizer
import torchvision
from torchvision import datasets, transforms

import trainer.ml as ml
from trainer.ml.data_loading import random_subject_generator
from trainer.lib import create_identifier
from trainer.ml.visualization import VisBoard

# If GPU is available, use GPU
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
IDENTIFIER = create_identifier()


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
                 ds: ml.Dataset,
                 batch_size=4):
        super().__init__()
        self.name = model_name
        self.model, self.optimizer, self.criterion, self.ds, self.batch_size = model, opti, crit, ds, batch_size
        self.model = self.model.to(device)
        self.gen = {
            '_all_': random_subject_generator(self.ds, self.preprocess, split='train', batchsize=self.batch_size)
        }

    def sample_minibatch(self, split='') -> Tuple[torch.Tensor, torch.Tensor]:
        if not split:
            x, y = next(self.gen['_all_'])
        else:
            if split not in self.gen:
                self.gen[split] = random_subject_generator(
                    self.ds,
                    self.preprocess,
                    split=split,
                    batchsize=self.batch_size)
            x, y = next(self.gen[split])

        return x.to(device), y.to(device)

    def train_on_minibatch(self, training_example: Tuple[torch.Tensor, torch.Tensor]) -> float:
        x, y = training_example

        self.optimizer.zero_grad()
        y_ = self.model(x)

        loss = self.criterion(y_, y)
        loss.backward()
        self.optimizer.step()

        batch_loss = loss.item()  # Loss, in the end, should be a single number
        return batch_loss

    @abstractmethod
    def preprocess(self, s: ml.Subject) -> Tuple[np.ndarray, np.ndarray]:
        """
        Provides the preprocessing chain to extract a training example from a subject.
        :param s: One subject
        :return: The training example (x, y), of type torch.Tensor
        """
        pass

    @abstractmethod
    def visualize_input_batch(self) -> plt.Figure:
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
# def compare_architectures(models: List[nn.Module], writer: VisBoard) -> List[int]:
#     import inspect
#
#     # Instantiate, because model.parameters does not work on the class definition
#     instantiated_list = []
#     for model in models:
#         if inspect.isclass(model):
#             instantiated_list.append(model())
#         else:
#             instantiated_list.append(model)
#     models = instantiated_list
#
#     params = [sum([p.numel() for p in model.parameters()]) for model in models]
#
#     fig, ax1 = plt.subplots()
#
#     ax1.set_title(f"Number of Parameters")
#     sns.barplot(x=[m.name for m in models], y=params, ax=ax1)
#
#     writer.add_figure(fig, group_name='Parameter Number')
#     return params
