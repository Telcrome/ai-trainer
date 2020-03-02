"""
This module contains the tooling for:

- CLI tools for training and long file/import/export operations.
"""

import os

import click
from tqdm import tqdm

import trainer.lib as lib
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
    d: lib.Dataset = lib.Session().query(lib.Dataset).filter(lib.Dataset.name == dataset_name).first()
    if d is None:
        print('There is no such dataset')
    if not subject_name:
        # Subject name needs to be picked
        subject_name = d.splits[0].sbjts[0]  # Just pick the first subject
    run_window(AnnotationGui, subject_name, dataset_name)


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
