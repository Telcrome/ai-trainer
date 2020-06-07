import os
import random
from abc import ABC, abstractmethod
from enum import Enum
from functools import partial
from typing import Tuple, Union, Callable, List, Iterator, Any

import numpy as np
import cv2

try:
    import imgaug.augmenters as iaa
    from imgaug.augmentables.segmaps import SegmentationMapsOnImage
except ImportError as _:
    pass
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
# device = torch.device('cpu')
IDENTIFIER = lib.create_identifier()


class ModelMode(Enum):
    """
    Used to differentiate what the model is currently doing.

    The following guidelines apply for the semantics of this enum:

    - Train asks for augmentation and other tricks during training (batch normalization, ...)
    - Eval does not require augmentation and is used for evaluation
    - Usage does not require ground truths
    """
    Train = "Train"
    Eval = "Eval"
    Usage = "Usage"


class InMemoryDataset(data.Dataset):
    """
    Wrapper around one dataset split to work with the torch.utils.data.Dataloader.
    This dataloader can be used to perform augmentations on multiple processes on the CPU and train on the GPU.
    """

    def __init__(self,
                 ds_name: str,
                 split_name: str,
                 # List[Tuple[List[np.ndarray], np.ndarray]]], partial],
                 f: Union[Callable[[lib.Subject, ModelMode], Any], partial],
                 mode: ModelMode = ModelMode.Train,
                 subject_filter: Union[Callable[[lib.Subject], bool], None] = None):
        super().__init__()
        self.preprocessor = f
        session = lib.Session()

        self.ds = session.query(lib.Dataset).filter(lib.Dataset.name == ds_name).first()
        self.split = session.query(lib.Split) \
            .filter(self.ds.id == lib.Split.dataset_id) \
            .filter(lib.Split.name == split_name) \
            .options(joinedload(lib.Split.sbjts)
                     .joinedload(lib.Subject.ims, innerjoin=True)
                     .joinedload(lib.ImStack.semseg_masks, innerjoin=True)) \
            .first()

        self.mode = mode
        self.subject_filter = subject_filter
        self.subjects = self.split.sbjts

        if self.subject_filter is not None:
            self.subjects = list(filter(self.subject_filter, self.subjects))

    def get_torch_dataloader(self, **kwargs):
        return data.DataLoader(self, **kwargs)

    def get_random_batch(self):
        return self.__getitem__(random.randint(0, self.__len__() - 1))

    def __getitem__(self, item) -> Tuple[np.ndarray, np.ndarray]:
        """
        Uses the preprocessor that converts a subject to a training example.

        :param item: Name of a subject
        :return: Training example x, y
        """
        s = self.subjects[item]

        t = self.preprocessor(s, self.mode)

        # Cannot transformed to cuda tensors at this point,
        # because they do not seem to work in shared memory. Return numpy arrays instead.
        return t

    def __len__(self):
        return len(self.subjects)


class SemSegDataset(data.Dataset):
    """
    Dataset Wrapper that simplifies the common case of performing single image semantic segmentation.
    Given a split it looks for all masks and yields pairs (image: np.ndarray, mask: np.ndarray).

    The dataset is loaded into memory, therefore your data has to be small enough.
    """

    def __init__(self,
                 ds_name: str,
                 split_name: str,
                 f: Union[Callable[[Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray]], partial] = None,
                 mode: ModelMode = ModelMode.Train,
                 session=lib.Session()):
        super().__init__()
        self.session = session
        self.preprocessor = f

        self.ds = session.query(lib.Dataset).filter(lib.Dataset.name == ds_name).first()
        self.split = session.query(lib.Split) \
            .filter(self.ds.id == lib.Split.dataset_id) \
            .filter(lib.Split.name == split_name) \
            .options(joinedload(lib.Split.sbjts)
                     .joinedload(lib.Subject.ims, innerjoin=True)
                     .joinedload(lib.ImStack.semseg_masks, innerjoin=True)) \
            .first()

        self.mode = mode

        self.masks: List[Tuple[np.ndarray, np.ndarray]] = []
        for sbjt in self.split.sbjts:
            for imstack in sbjt.ims:
                for ssmask in imstack.semseg_masks:
                    f_number = ssmask.for_frame
                    self.masks.append((imstack.get_ndarray()[f_number], ssmask.get_ndarray()))

    def export_to_dir(self, dir_name, model: nn.Module):
        os.mkdir(dir_name)
        for sbjt in self.split.sbjts:
            print(f'Exporting {sbjt.name} to {dir_name}')
            sbjt_folder = os.path.join(dir_name, sbjt.name)
            os.mkdir(sbjt_folder)
            for i, imstack in enumerate(sbjt.ims):
                imstack_folder = os.path.join(sbjt_folder, f'imstack{i}')
                os.mkdir(imstack_folder)
                frames_with_mask = {ssmask.for_frame: ssmask for ssmask in imstack.semseg_masks}
                for frame_id in range(imstack.get_ndarray().shape[0]):
                    im_arr = imstack.get_ndarray()[frame_id]
                    model_input_size = ml.normalize_im(cv2.resize(im_arr, (384, 384)))
                    model_input = torch.from_numpy(np.rollaxis(model_input_size, 2, 0).astype(np.float32)).unsqueeze(0)
                    pred_arr = torch.sigmoid(model(model_input)).detach().numpy()[0]
                    for class_id in range(pred_arr.shape[0]):
                        class_arr = pred_arr[class_id]
                        cv2.imwrite(os.path.join(imstack_folder, f'{frame_id}class{class_id}.png'), class_arr * 255)
                    true_arr = frames_with_mask[frame_id].get_ndarray() if frame_id in frames_with_mask else None
                    # if true_arr is not None:
                    #     for class_id in range(true_arr.shape[2]):
                    #         cv2.imwrite(os.path.join(imstack_folder, f'{frame_id}gt{class_id}.png'), true_arr[:, :, class_id])
                    cv2.imwrite(os.path.join(imstack_folder, f'{frame_id}image.png'), im_arr)

    @staticmethod
    def aug_preprocessor(t: Tuple[np.ndarray, np.ndarray]):
        im, gt = t
        seq = iaa.Sequential([
            iaa.Dropout([0.01, 0.2]),  # drop 5% or 20% of all pixels
            iaa.Crop(percent=(0, 0.1)),
            iaa.Fliplr(0.5),
            iaa.Sharpen((0.0, 1.0)),  # sharpen the image
            # iaa.SaltAndPepper(0.1),
            iaa.WithColorspace(
                to_colorspace="HSV",
                from_colorspace="RGB",
                children=iaa.WithChannels(
                    0,
                    iaa.Add((0, 50))
                )
            ),
            iaa.Sometimes(p=0.5, then_list=[iaa.Affine(rotate=(-10, 10))])
            # iaa.ElasticTransformation(alpha=50, sigma=5)  # apply water effect (affects segmaps)
        ], random_order=True)
        segmap = SegmentationMapsOnImage(gt, shape=im.shape)
        im, gt = seq(image=im, segmentation_maps=segmap)
        gt = gt.arr
        return im, gt

    def get_torch_dataloader(self, **kwargs):
        return data.DataLoader(self, **kwargs)

    def get_random_batch(self):
        return self.__getitem__(random.randint(0, self.__len__() - 1))

    def __getitem__(self, item) -> Tuple[np.ndarray, np.ndarray]:
        """
        Uses the preprocessor that converts a subject to a training example.

        :param item: Index of a training example
        :return: Training example x, y
        """
        x, y = self.masks[item]

        if x.shape[2] == 1:
            x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)

        x = cv2.resize(x, (384, 384))
        y = cv2.resize(y.astype(np.uint8), (384, 384)).astype(np.bool)

        if self.preprocessor is not None:
            x, y = self.preprocessor((x, y))
            y = y.astype(np.bool)  # Will later be used for indexing, therefore needs to be boolean

        x = np.rollaxis(ml.normalize_im(x), 2, 0)
        y = np.rollaxis(y, 2, 0)
        gt = np.zeros((y.shape[1], y.shape[2]), dtype=np.int)
        for c_id in range(y.shape[0]):
            gt[y[c_id]] = c_id + 1
        # y = np.argmax(y, axis=0)

        # gt = np.zeros((y.shape[1], y.shape[2]))
        # gt[y[0]] = 1
        # gt[y[1]] = 2

        # Cannot transform to cuda tensors at this point,
        # because they do not seem to work in shared memory. Return numpy arrays instead.
        # return torch.from_numpy(x), torch.from_numpy(gt).long()
        return x.astype(np.float32), gt

    def __len__(self):
        return len(self.masks)


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
        raise Exception("The model is not initialized")

    return sum([p.numel() for p in model.parameters()])


def plot_grad_flow(named_parameters: Iterator[Tuple[str, torch.nn.Parameter]]) -> plt.Figure:
    """
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    """
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n):
            layers.append(n.replace('.weight', ''))
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    ax.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    ax.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    ax.set_xticks(range(0, len(ave_grads), 1))
    ax.set_xticklabels(layers, rotation=45)
    ax.set_xlim(left=0, right=len(ave_grads))
    ax.set_ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    ax.set_xlabel("Layers")
    ax.set_ylabel("average gradient")
    ax.set_title("Gradient flow")
    ax.grid(True)
    ax.legend([Line2D([0], [0], color="c", lw=4),
               Line2D([0], [0], color="b", lw=4),
               Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    return fig


class ModelTrainer:
    """
    TrainerModel is the user of a torch nn.Module model and implements common training and evaluation methods.
    """

    def __init__(self,
                 exp_name: str,
                 model: nn.Module,
                 opti: optimizer.Optimizer,
                 crit: Union[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], nn.Module],
                 weights_initializer=init_weights,
                 train_batch_callback: Callable[[], None] = None):
        self.model_name = exp_name
        self.model, self.optimizer, self.criterion = model, opti, crit
        self.model = self.model.to(device)
        self.weights_initializer = weights_initializer
        if train_batch_callback is not None:
            self.train_batch_callback = train_batch_callback
        else:
            self.train_batch_callback = lambda: True

    def print_summary(self):
        print(f"Capacity of the network: {get_capacity(self.model)}")
        # ml.logger.add_model(self.model, input_batch[0].unsqueeze(0).to(ml.torch_device))

    def init_weights(self):
        self.model.apply(init_weights)

    def train_minibatch(self,
                        training_examples: Tuple[torch.Tensor, torch.Tensor],
                        mode: ModelMode,
                        visu: Callable[[int, ModelMode, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]], None] = None,
                        evaluator: TrainerMetric = None,
                        epoch=-1) -> float:
        self.train_batch_callback()
        # noinspection PyArgumentList
        loss = torch.Tensor([0.]).to(device)
        self.optimizer.zero_grad()

        vis_ls = []

        # inps, y = training_example
        # inps, y = [inp.to(device) for inp in inps], y.to(device)
        x, y = training_examples[0].to(device), training_examples[1].to(device)

        y_ = self.model(x)
        # if visu is not None:
        #     vis_ls.append(([a.detach().cpu().numpy() for a in inps],
        #                    y.detach().cpu().numpy(),
        #                    y_.detach().cpu().numpy()))

        seq_item_loss = self.criterion(y_, y)
        if seq_item_loss < 0:
            print("Loss smaller 0 is not possible")
        loss += seq_item_loss

        if mode == ModelMode.Train:
            loss.backward()
            self.optimizer.step()
        if evaluator is not None:
            # noinspection PyUnboundLocalVariable
            evaluator.update(y_.detach().cpu().numpy(), y.detach().cpu().numpy())

        if visu is not None and random.random() > 0.99:
            visu(epoch, mode, vis_ls)
            ml.logger.visboard.add_figure(plot_grad_flow(self.model.named_parameters()), group_name='gradient_analysis')

        batch_loss = loss.item()  # Loss, in the end, should be a single number

        return batch_loss

    def run_epoch(self, torch_loader: data.DataLoader,
                  epoch: int,
                  batch_size: int,
                  mode: ModelMode,
                  steps=-1,
                  visu: Callable[[int, ModelMode, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]], None] = None,
                  metric: TrainerMetric = None):
        if mode == ModelMode.Train:
            self.model.train()
        else:
            self.model.eval()
        epoch_loss_sum = 0.

        steps = len(torch_loader) if steps == -1 else steps
        lib.logger._log_str(f'Starting epoch: {epoch} with N={len(torch_loader) * batch_size} and {steps} steps\n')
        loader_iter = iter(torch_loader)
        with tqdm(total=steps, maxinterval=steps / 100) as pbar:
            for i in range(steps):
                try:
                    seq = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(torch_loader)
                    seq = next(loader_iter)

                torch_env = torch.no_grad if mode == ModelMode.Eval else torch.enable_grad
                with torch_env():
                    loss = self.train_minibatch(seq, mode=mode, evaluator=metric, epoch=epoch, visu=visu)

                # Log metrics and loss
                epoch_loss_sum += (loss / batch_size)
                epoch_loss = epoch_loss_sum / (i + 1)
                lib.logger.add_scalar(f'{mode.value}: loss/train epoch {epoch + 1}', epoch_loss, i)

                # Handle progress bar
                pbar.update()
                display_loss = epoch_loss_sum / (i + 1)
                orientation_str = f"Epoch: {epoch}, Mode: {mode.value}"
                if metric is not None:
                    pbar.set_description(
                        f'{orientation_str}, Loss: {display_loss:05f}, Metric: {metric.get_result():05f}')
                else:
                    pbar.set_description(
                        f'{orientation_str}, Loss: {display_loss:05f}')
        lib.logger.debug_var(f"\n{mode.value} epoch result: {epoch_loss_sum / steps}\n")

    def save_to_disk(self, dir_path: str = '.', hint=''):
        torch.save(self.model.state_dict(), os.path.join(dir_path, f'{self.model_name}{hint}.pt'))

    def load_from_disk(self, f_path: str) -> bool:
        p = f_path
        if os.path.exists(p):
            self.model.load_state_dict(torch.load(p))
            return True
        else:
            print(f"{self.model} not yet on disk, train first.")
            return False
