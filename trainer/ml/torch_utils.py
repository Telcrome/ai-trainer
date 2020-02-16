import os
from abc import ABC, abstractmethod
from enum import Enum
from functools import partial
from typing import Tuple, Union, Callable, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sqlalchemy.orm import joinedload
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


class InMemoryDataset(data.Dataset):
    """
    Wrapper around one dataset split to work with the torch.utils.data.Dataloader.
    This dataloader can be used to perform augmentations on multiple processes on the CPU and train on the GPU.
    """

    def __init__(self,
                 ds_name: str,
                 split_name: str,
                 f: Union[Callable[[lib.Subject, ModelMode], List[Tuple[List[np.ndarray], np.ndarray]]], partial],
                 mode: ModelMode = ModelMode.Train):
        super().__init__()
        self.preprocessor = f
        session = lib.Session()

        self.ds = session.query(lib.Dataset).filter(lib.Dataset.name == ds_name).first()
        self.split = session.query(lib.Split) \
            .options(joinedload(lib.Split.sbjts)
                     .joinedload(lib.Subject.ims, innerjoin=True)
                     .joinedload(lib.ImStack.semseg_masks, innerjoin=True)) \
            .filter(self.ds.id == lib.Split.dataset_id and lib.Split.name == split_name) \
            .first()

        self.mode = mode

    def get_torch_dataloader(self, **kwargs):
        return data.DataLoader(self, **kwargs)

    def __getitem__(self, item) -> Tuple[np.ndarray, np.ndarray]:
        """
        Uses the preprocessor that converts a subject to a training example.

        :param item: Name of a subject
        :return: Training example x, y
        """
        s = self.split.sbjts[item]
        # if not self.in_memory:
        #     self.session.add(s)
        t = self.preprocessor(s, self.mode)

        # Cannot transformed to cuda tensors at this point,
        # because they do not seem to work in shared memory. Return numpy arrays instead.
        return t

    def __len__(self):
        return len(self.split.sbjts)


def bench_mark_dataset(ds: InMemoryDataset, extractor: Callable):
    res = []
    with tqdm(total=len(ds), maxinterval=len(ds) / 100) as pbar:
        for i in range(len(ds)):
            s = ds.__getitem__(i)
            res.append(extractor(s))
            pbar.update()
    return res


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
            prediction = np.argmax(prediction, axis=1)
            if type(prediction) != np.ndarray:
                prediction = np.array(prediction)

        self.preds.extend(list(prediction))
        self.targets.extend(list(target))

    def get_result(self):
        return accuracy_score(self.targets, self.preds)


def init_weights(layer: nn.Module) -> None:
    if isinstance(layer, nn.Linear):
        # Visualize a fully connected layer
        nn.init.xavier_uniform_(layer.weight)
        # nn.init.xavier_uniform(layer.bias)
    elif isinstance(layer, nn.Conv2d):
        # Visualize a convolutional layer
        nn.init.xavier_uniform_(layer.weight)
        # nn.init.xavier_uniform_(layer.bias)


def get_capacity(model: nn.Module) -> int:
    """
    Computes the number of parameters of a network.
    """
    import inspect

    # Instantiate, because model.parameters does not work on the class definition
    if inspect.isclass(model):
        model = model()

    return sum([p.numel() for p in model.parameters()])


class TrainerModel(ABC):
    """
    TrainerModel is the user of a torch nn.Module model and implements common training and evaluation methods.
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
        self.init_weights()

    def print_summary(self):
        print(f"Capacity of the network: {get_capacity(self.model)}")

    def init_weights(self):
        self.model.apply(init_weights)

    def init_hidden(self) -> List[torch.Tensor]:
        raise NotImplementedError()

    def train_on_minibatch(self,
                           training_examples: List[Tuple[List[torch.Tensor], torch.Tensor]],
                           evaluator: TrainerMetric = None) -> float:
        hidden_states = self.init_hidden()
        # noinspection PyArgumentList
        loss = torch.Tensor([0.]).to(device)
        self.optimizer.zero_grad()
        for training_example in training_examples:
            inps, y = training_example
            inps, y = [inp.to(device) for inp in inps], y.to(device)

            y_, hidden_states = self.model.forward(inps, hidden_states)

            seq_item_loss = self.criterion(y_, y)
            loss += seq_item_loss
        loss.backward()
        self.optimizer.step()
        # noinspection PyUnboundLocalVariable
        evaluator.update(y_.detach().cpu().numpy(), y.detach().cpu().numpy())

        batch_loss = loss.item()  # Loss, in the end, should be a single number
        return batch_loss

    def handle_minibatch(self, seq: List[Tuple[List[torch.Tensor], torch.Tensor]], metric: TrainerMetric = None):
        return self.train_on_minibatch(seq, evaluator=metric)

    def run_epoch(self, torch_loader: data.DataLoader, epoch: int, batch_size: int, steps=-1,
                  metric: TrainerMetric = None):
        self.model.train()
        epoch_loss_sum = 0.

        steps = len(torch_loader) if steps == -1 else steps
        ml.logger.log(
            f'Starting epoch: {epoch} with {len(torch_loader) * batch_size} training examples and {steps} steps\n')
        loader_iter = iter(torch_loader)
        with tqdm(total=steps, maxinterval=steps / 100) as pbar:
            for i in range(steps):
                seq = next(loader_iter)

                loss = self.handle_minibatch(seq, metric=metric)

                # Log metrics and loss
                epoch_loss_sum += (loss / batch_size)
                epoch_loss = epoch_loss_sum / (i + 1)
                ml.logger.add_scalar(f'loss/train epoch {epoch + 1}', epoch_loss, i)

                # Handle progress bar
                pbar.update()
                display_loss = epoch_loss_sum / (i + 1)
                pbar.set_description(f'Loss: {display_loss:05f}, Metric: {metric.get_result()}')
        ml.logger.log(f"\nEpoch result: {epoch_loss_sum / steps}\n")

    def evaluate(self, eval_loader: data.DataLoader, evaluator: TrainerMetric):
        self.model.eval()
        steps = len(eval_loader)
        eval_iter = iter(eval_loader)
        with torch.no_grad():  # Testing does not require gradients
            with tqdm(total=steps, maxinterval=steps / 100) as pbar:
                for i in range(steps):
                    ts = next(eval_iter)

                    hs = self.init_hidden()
                    for t in ts:
                        x, y = t

                        ml.logger.log(f'({x[0][0, 0]}, {x[0][0, 1]}) -> ({y[0, 0]}, {y[0, 1]})')
                        inps = [inp.to(device) for inp in x]
                        y_, hs = self.model.forward(inps, hs)

                        y, y_ = y.numpy(), y_.cpu().numpy()
                        ml.logger.log(f'Prediction: {({y_[0, 0]}, {y_[0, 1]})}')
                    evaluator.update(y_, y)

                    pbar.update()
                    pbar.set_description(f"{evaluator.get_result():05f}")
                    ml.logger.log('\n\n')

    def save_to_disk(self, dir_path: str = '.'):
        torch.save(self.model.state_dict(), os.path.join(dir_path, f'{self.model_name}.pt'))

    def load_from_disk(self, dir_path: str = '.') -> bool:
        p = os.path.join(dir_path, f'{self.model_name}.pt')
        if os.path.exists(p):
            self.model.load_state_dict(torch.load(p))
            return True
        else:
            print(f"{self.model} not yet on disk, train first.")
            return False
