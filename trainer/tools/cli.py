"""
This module contains the tooling for:

- CLI tools for training and long file/import/export operations.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import click
from tqdm import tqdm
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage

from ignite.contrib.handlers import ProgressBar

import trainer.lib as lib
import trainer.ml as ml
from trainer.tools.AnnotationGui import AnnotationGui, run_window


@click.group()
def trainer():
    """
    AI command line tools.
    """
    pass


@trainer.command(name='list-subjects')
def trainer_list_subjects():
    res = lib.Session().query(lib.Subject)  # @.filter(lib.Dataset.name == dataset_name)
    for s in res:
        print(s)


@trainer.command(name='list-datasets')
def trainer_list_subjects():
    res = lib.Session().query(lib.Dataset)  # @.filter(lib.Dataset.name == dataset_name)
    for d in res:
        print(d)


@trainer.command(name='init')
@click.option('--parent-path', '-p', default=os.getcwd, help='Directory that the dataset will appear in')
@click.option('--name', '-n', prompt=True, help='Name of the dataset created')
def dataset_init(parent_path, name):
    """
    Create a new dataset
    """
    ls = os.listdir(parent_path)
    # click.echo(f"Other datasets in {parent_path}")
    # for p in ls:
    #     if os.path.isdir(p):
    #         click.echo(f"Dirname: {os.path.basename(p)}")
    if click.confirm(f"The dataset {name} will be created in {parent_path}"):
        d = lib.Dataset.build_new(name, parent_path)
        d.to_disk(parent_path)
    click.echo(f"For working with the dataset {name}, please switch into the directory")


@trainer.group()
def ds():
    """"
    Command line tools concerned with one dataset
    """
    pass


@ds.command(name="annotate")
@click.option('--dataset-name', '-n', prompt='Dataset Name:', help='Name of the dataset')
@click.option('--subject-name', '-s', default='', help='If provided, opens the given subject from the dataset')
def dataset_annotate(dataset_name: str, subject_name: str):
    """
    Start annotating subjects in the dataset.
    """
    sess = lib.Session()
    d: lib.Dataset = sess.query(lib.Dataset).filter(lib.Dataset.name == dataset_name).first()
    if d is None:
        print('There is no such dataset')
    if not subject_name:
        # Subject name needs to be picked
        s = d.splits[0].sbjts[0]  # Just pick the first subject
    else:
        raise NotImplementedError()
    run_window(AnnotationGui, d, s, sess)


@ds.command(name="export-predictions")
@click.option('--dataset-name', '-n', prompt='Dataset Name', help='Name of the dataset')
@click.option('--split-name', '-s', prompt='Split Name', help='Name of the dataset split')
@click.option('--weights-path', '-w', help='Path to the weights of the network')
def export_predictions(dataset_name: str, split_name: str, weights_path: str):
    """
    Export the dataset to disk using the model for autocompletion.

    :param dataset_name:
    :param split_name:
    :param weights_path:
    :return:
    """
    model = smp.PAN(in_channels=3, classes=3)
    model.eval()
    model.load_state_dict(torch.load(weights_path))
    dataset = ml.SemSegDataset(dataset_name, split_name, mode=ml.ModelMode.Usage)
    dataset.export_to_dir(os.path.join(os.getcwd(), lib.create_identifier('export')), model)


@ds.command(name='export-all')
@click.option('--export-folder', '-p', default=os.getcwd())
@click.option('--data-split', '-s', prompt='Enter name of the data split to be exported')
def dataset_export_all(export_folder: str, data_split: str):
    split = lib.Session().query(lib.Split).filter(lib.Split.name == data_split).first()
    lib.export_to_folder(split, export_folder)


@ds.command(name="train")
@click.option('--dataset-name', '-n', prompt='Dataset Name:', help='Name of the dataset')
@click.option('--split-name', '-sn', prompt='Split Name:', help='Name of the training split')
@click.option('--weights-path', '-w', help='A starting point for the learnable model parameters', default='')
@click.option('--target-path', '-w', help='Path where the model weights are saved', default='')
@click.option('--batch-size', default=4, help='Batch Size for training and evaluation')
@click.option('--epochs', default=50, help='Epochs: One training pass through the training data')
@click.option('--eval-split', default='', help='Split that the model is evaluated on')
def dataset_train(dataset_name: str, split_name: str, weights_path: str, target_path: str, batch_size: int,
                  epochs: int, eval_split: str):
    """
    Start annotating subjects in the dataset.
    """
    # if not weights_path:
    #     weights_path, _ = lib.standalone_foldergrab(folder_not_file=False)
    if not target_path:
        target_path, _ = lib.standalone_foldergrab(folder_not_file=True)
    if not eval_split:
        eval_split = split_name

    train_set = ml.SemSegDataset(dataset_name, split_name, f=ml.SemSegDataset.aug_preprocessor, mode=ml.ModelMode.Train)
    eval_set = ml.SemSegDataset(dataset_name, eval_split, mode=ml.ModelMode.Eval)

    train_loader = train_set.get_torch_dataloader(batch_size=batch_size, shuffle=True)
    eval_loader = eval_set.get_torch_dataloader(batch_size=batch_size, shuffle=True)

    def vis(inps: np.ndarray, preds: np.ndarray, targets: np.ndarray) -> None:
        for batch_id in range(inps.shape[0]):
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
            sns.heatmap(inps[batch_id, 0, :, :], ax=ax1)
            sns.heatmap(targets[batch_id, :, :], ax=ax2)
            # sns.heatmap(preds[batch_id, 0, :, :], ax=ax2)
            sns.heatmap(preds[batch_id, 1, :, :], ax=ax3)
            sns.heatmap(preds[batch_id, 2, :, :], ax=ax4)
            fig.show()

    model = smp.PAN(in_channels=3, classes=3)
    opti = optim.Adam(model.parameters(), lr=5e-3)
    crit = ml.SegCrit(1., 2., (1.0, 0.5))

    net = ml.ModelTrainer(
        lib.create_identifier('PAN'),
        model,
        opti,
        crit
    )
    if weights_path:
        net.load_from_disk(weights_path)

    def vis_loader(loader):
        with torch.no_grad():
            b = next(iter(loader))
            x, y = b
            # ml.SegNetwork.visualize_input_batch(b)
            out = model(x.to(ml.torch_device))
            out = torch.sigmoid(out)
            vis(x.numpy(), out.cpu().numpy(), y.numpy())

    for epoch in range(epochs):
        vis_loader(train_loader)
        net.run_epoch(train_loader, epoch=epoch, mode=ml.ModelMode.Train, batch_size=batch_size)
        net.save_to_disk(target_path, hint=f'{epoch}')
        vis_loader(eval_loader)


@ds.command(name='add-image-folder')
@click.option('--dataset-path', '-p', default=os.getcwd)
@click.option('--folder-path', '-ip', default='')
@click.option('--structure-tpl', '-st', default='')
def dataset_add_image_folder(dataset_path: str, folder_path: str, structure_tpl: str):
    d = lib.Dataset.from_disk(dataset_path)
    if not folder_path:
        folder_path, inputs_dict = lib.standalone_foldergrab(
            folder_not_file=True,
            title='Pick Image folder',
            optional_choices=[('Structure Template', 'str_tpl', d.get_structure_template_names())]
        )
        structure_tpl = inputs_dict['str_tpl']
    seg_structs = d.get_structure_template_by_name(structure_tpl)
    lib.add_image_folder(d, folder_path, structures=seg_structs)


@ds.command(name='add-ml-folder')
@click.option('--dataset-path', '-p', default=os.getcwd)
@click.option('--folder-path', '-ip', default='')
@click.option('--structure-tpl', '-st', default='')
def dataset_add_ml_folder(dataset_path: str, folder_path: str, structure_tpl: str):
    """
    Imports a computer vision related folder into the trainer format.
    Currently supports:
    - Images with segmentation masks

    Assumes a folder structure of the following form:

    - train
        - im (training images)
            - single_image.jpg
            - subject_folder
                - one.jpg
                - ...
        - gt_humans (binary segmentation maps for class humans)
            - single_image.jpg
            - subject_folder
                - one.jpg
                - ...
        - gt_cars (segmentation maps for class cars)
        - ...
    - test
        - ...

    The name of the source image and its corresponding ground truths must be identical.
    The structure template must exist beforehand and must contain the knowledge about the given supervised data.
    """
    d = lib.Dataset.from_disk(dataset_path)
    if not folder_path:
        folder_path, inputs_dict = lib.standalone_foldergrab(
            folder_not_file=True,
            title='Pick Image folder',
            optional_choices=[('Structure Template', 'str_tpl', d.get_structure_template_names())]
        )
        structure_tpl = inputs_dict['str_tpl']
    seg_structs = d.get_structure_template_by_name(structure_tpl)

    # Iterate over splits (top-level-directories)
    for split in filter(os.path.isdir, [os.path.join(folder_path, fn) for fn in os.listdir(folder_path)]):
        ims_folder = os.path.join(split, 'im')
        for path_dir, path_name in tqdm([(os.path.join(ims_folder, fn), fn) for fn in os.listdir(ims_folder)]):
            # Compute the ground truths
            gt_folders = [(os.path.join(split, p), p) for p in os.listdir(split) if p != 'im']
            lib.import_utils.append_subject(
                d,
                (path_dir, path_name),
                gt_folders,
                seg_structs,
                split=os.path.split(split)[-1])
        # lib.import_utils.add_to_split(d, dicts)
    d.to_disk()


if __name__ == '__main__':
    trainer()
