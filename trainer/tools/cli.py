"""
This module contains the tooling for:

- CLI tools for training and long file/import/export operations.
"""

import os

import click
from tqdm import tqdm

from trainer.lib import standalone_foldergrab
from trainer.lib.JsonClass import dir_is_json_class
from trainer.tools.AnnotationGui import AnnotationGui, run_window
import trainer.ml as ml
import trainer.lib as lib


@click.group()
def trainer():
    """
    AI command line tools.
    """
    pass


@trainer.command(name='download')
@click.option('--parent-path', '-p', default=os.getcwd, help='Directory that the dataset will appear in')
@click.option('--name', '-n', help='If provided, the dataset will not be redownloaded everytime')
@click.option('--url', '-u', help='Needs to point to a zip file')
def dataset_download(parent_path, name, url):
    from trainer.lib import download_and_extract
    local_path = download_and_extract(online_url=url,
                                      parent_dir=parent_path,
                                      dir_name=name)
    output = f'The directory is saved to {local_path}'
    click.echo(output)


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
        d = ml.Dataset.build_new(name, parent_path)
        d.to_disk(parent_path)
    click.echo(f"For working with the dataset {name}, please switch into the directory")


@trainer.group()
def ds():
    """"
    Command line tools concerned with one dataset that is currently on pwd
    """
    pass


@ds.command(name="annotate")
@click.option('--dataset-path', '-p', default=os.getcwd, help='Path to a dataset')
@click.option('--subject-name', '-s', default='', help='If provided, opens the given subject from the dataset')
def dataset_annotate(dataset_path: str, subject_name: str):
    """
    Start annotating subjects in the dataset.
    """
    if not subject_name:
        # Subject name needs to be picked
        d = ml.Dataset.from_disk(dataset_path)
        subject_name = d.get_subject_name_list()[0]  # Just pick the first subject
    run_window(AnnotationGui, os.path.join(dataset_path, subject_name), dataset_path)


@ds.command(name="server")
@click.option('--dataset-path', '-p', default=os.getcwd, help='Path to a dataset')
def dataset_serve(dataset_path: str):
    print(dataset_path)
    from trainer.server import app
    app.run()


@ds.command(name="train")
@click.option('--dataset-path', '-p', default=os.getcwd, help='The path to the directory where the dataset lives')
def dataset_train(dataset_path: str):
    """
    Trains the models that can be trained from this dataset.

    While True:
        1. Analyses the current subjects for predictable content (structures, classes...)
        2. Determines which models can be trained for helping the annotation process
        3. Cycles over the different models:
            3.1 -> trains them for one epoch
            3.2 -> saves the weights and metadata into the dataset.
    """
    if not dir_is_json_class(dataset_path):
        raise Exception("The given directory is not a valid Dataset")
    d = ml.Dataset.from_disk(dataset_path)

    seg_structs = d.compute_segmentation_structures()

    for str_name in seg_structs:
        click.echo(f"\n{str_name}: {len(seg_structs[str_name])}")
        from trainer.ml.predictor import compile_and_train
        compile_and_train(d, str_name)


@ds.command(name='visualize')
@click.option('--dataset-path', '-p', default=os.getcwd)
@click.option('--subject-name', '-s', default='')
def dataset_visualize(dataset_path: str, subject_name: str):
    d = ml.Dataset.from_disk(dataset_path)
    if subject_name:
        s = d.get_subject_by_name(subject_name)
        s.matplot_imagestacks()
    else:
        # TODO Create a html folder with a simple static webview
        # TODO Include Media and Metadata
        raise NotImplementedError()


@ds.command(name='add-image-folder')
@click.option('--dataset-path', '-p', default=os.getcwd)
@click.option('--folder-path', '-ip', default='')
@click.option('--structure-tpl', '-st', default='')
def dataset_add_image_folder(dataset_path: str, folder_path: str, structure_tpl: str):
    d = ml.Dataset.from_disk(dataset_path)
    if not folder_path:
        folder_path, inputs_dict = standalone_foldergrab(
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
    d = ml.Dataset.from_disk(dataset_path)
    if not folder_path:
        folder_path, inputs_dict = standalone_foldergrab(
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
