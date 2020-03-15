"""
This module contains the tooling for:

- CLI tools for training and long file/import/export operations.
"""

import os
from typing import List

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

import trainer.lib as lib
import trainer.ml as ml
from trainer.tools.AnnotationGui import AnnotationGui, run_window


@click.group()
def trainer():
    """
    AI command line tools.
    """
    pass


@trainer.command(name='reset-database')
def trainer_reset_database():
    """
    Removes all tables from the database and clears the big binary directory if it exists.
    """
    lib.reset_database()


@trainer.command(name='print-summary')
def trainer_print_summary():
    sess = lib.Session()

    print("### Semantic Segmentation Templates ###")
    ss_tpls: List[lib.SemSegTpl] = sess.query(lib.SemSegTpl).all()
    for ss_tpl in ss_tpls:
        print(f"Semantic segmentation template {ss_tpl.name} contains the following classes:")
        for ss_class in ss_tpl.ss_classes:
            print(f"Semantic Segmentation class {ss_class.name} of type {ss_class.ss_type}")

    print("### Class Definitions ###")
    cls_defs: List[lib.ClassDefinition] = sess.query(lib.ClassDefinition).all()
    for cls_def in cls_defs:
        print(cls_def)

    print("### Dataset Summaries ###")
    dss: List[lib.Dataset] = sess.query(lib.Dataset).all()
    for ds in dss:
        print(ds.get_summary())


@trainer.command(name='add-semseg-tpl')
@click.option('--tpl-name', '-n', prompt='Enter name of semantic segmentation template')
def trainer_add_semseg_tpl(tpl_name: str):
    cls_type_mapper = {
        'line': lib.MaskType.Line,
        'blob': lib.MaskType.Blob,
        'point': lib.MaskType.Point
    }

    sess = lib.Session()
    ss_types = {}
    while click.prompt("Enter other classes", type=click.Choice(['yes', 'no'], case_sensitive=False)) == 'yes':
        ss_cls_name = click.prompt('Name of the segmentation class')
        ss_cls_type = click.prompt(
            'Type of the segmentation class',
            type=click.Choice(cls_type_mapper.keys(), case_sensitive=False)
        )
        ss_types[ss_cls_name] = cls_type_mapper[ss_cls_type]
    if ss_types:
        print(f"Creating semantic segmentation template {tpl_name} with classes: \n {ss_types}")
        ss_tpl = lib.SemSegTpl.build_new(tpl_name, ss_types)
        sess.add(ss_tpl)
        sess.commit()
    else:
        print(f"{ss_types} not valid")


@trainer.command(name='add-class-def')
@click.option('--cls-name', '-n', prompt='Class Name', help='Name of the class')
@click.option('--cls-type', '-t', prompt='Class Type',
              type=click.Choice([e.value for e in lib.ClassType], case_sensitive=True))
def add_class_def(cls_name: str, cls_type: lib.ClassType):
    sess = lib.Session()
    cls_type_instance = lib.make_converter_dict_for_enum(lib.ClassType)[cls_type]

    vals: List[str] = []
    while True:
        user_choice = click.prompt("Enter class name, empty space if finished")
        if user_choice == ' ':
            break
        else:
            vals.append(user_choice)

    if not vals:
        raise Exception("Please provide class names")

    cls_def = lib.ClassDefinition.build_new(
        cls_name,
        cls_type=cls_type_instance,
        values=vals
    )
    sess.add(cls_def)
    sess.commit()


@trainer.command(name='list-subjects')
def trainer_list_subjects():
    res = lib.Session().query(lib.Subject)  # @.filter(lib.Dataset.name == dataset_name)
    for s in res:
        print(s)


@trainer.command(name='list-datasets')
def trainer_list_datasets():
    res = lib.Session().query(lib.Dataset)  # @.filter(lib.Dataset.name == dataset_name)
    for d in res:
        print(d)


@trainer.command(name='import')
@click.option('--dataset-name', '-d', prompt='Dataset name')
@click.option('--split-name', '-sn', prompt='How should the split for the imported data be called?')
@click.option('--folder-path', '-p', default=os.getcwd())
@click.option('--tpl-name', '-tpl', prompt='Name of the semantic segmentation template of the masks?')
def trainer_import(dataset_name: str, split_name: str, folder_path: str, tpl_name: str):
    """
    Imports data from a folder of the following structure:
    - split
      - subject1
        - imagestack1
          - im.npy (The actual image data)
          - 1.npy (mask for frame 1)
      ...
    """
    sess = lib.Session()
    d: lib.Dataset = sess.query(lib.Dataset).filter(lib.Dataset.name == dataset_name).first()
    semseg_tpl = sess.query(lib.SemSegTpl).filter(lib.SemSegTpl.name == tpl_name).first()
    split = d.add_split(split_name)
    lib.add_import_folder(split, folder_path, semseg_tpl)
    sess.commit()


@trainer.command(name="annotate")
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


@trainer.command(name="export-predictions")
@click.option('--dataset-name', '-ds', prompt='Dataset Name', help='Name of the dataset')
@click.option('--split-name', '-sp', prompt='Split Name', help='Name of the dataset split')
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


@trainer.command(name='export-all')
@click.option('--export-folder', '-p', default=os.getcwd())
@click.option('--data-split', '-s', prompt='Enter name of the data split to be exported')
def dataset_export_all(export_folder: str, data_split: str):
    split = lib.Session().query(lib.Split).filter(lib.Split.name == data_split).first()
    lib.export_to_folder(split, export_folder)


@trainer.command(name="train")
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


if __name__ == '__main__':
    trainer()
