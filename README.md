[Documentation](https://ai-trainer.readthedocs.io/en/latest/)

# Installation for User

Open anaconda powershell, activate an environment with anaconda, navigate into the trainer repo and execute the following to install trainer using pip, including its dependencies:

```shell script
pip install ai-trainer
```

For Online Learning you have to install PyTorch:
```shell script
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

AI-Trainer helps with building a data generator and it relies on imgaug for it:

```shell script
conda install imgaug -c conda-forge
```

For PostgreSQL connection trainer expects an environment variable "DB_CON" of the form:

```shell script
DB_PASS@127.0.0.1:5432/DB_NAME
```

# Getting started with training models

Trainer currently supports annotating images and videos.
First, create a dataset using

```shell script
trainer init-ds
cd YOUR_DATASET
```

# Getting started with using trainer in python

For using the annotated data, you can use trainer as a python package.
After activating the environment containing the trainer and its dependencies,
feel free to inspect some of the tutorials in ```./tutorials/```.

# Development Setup

Execute the user installation,
but instead of using `pip install ai-trainer`,
clone the repo locally.

```shell script
git clone https://github.com/Telcrome/ai-trainer
```

Both vsc and pycharm are used for development with
their configurations provided in ```.vscode``` and ```.idea```

## Recommended environments

For development we recommend to install the conda environment into a subfolder of the repo.
This allows for easier experimentation and the IDE expects it this way.

```shell script
conda env create --prefix ./envs -f environment.yml
conda activate .\envs\.
```

Now install a deep learning backend.
PyTorch provides well-working [conda install commands](https://pytorch.org/get-started/locally/).

For Tensorflow with GPU:
```shell script
conda install cudatoolkit=10.0 cudnn=7.6.0=cuda10.0_0
pip install tensorflow-gpu
```

### Testing Development for pip and cli tools

Installing the folder directly using pip does not work due to the large amount of files inside the local development folder,
especially because in the local development setup the environment is expected to be a subfolder of the repo.
```shell script
pip install -e .
```

# Using Docker

Docker and the provided DOCKERFILE support is currently experimental as it proved to slow down the annotation GUI too much.
When the transition to a web GUI is completed docker will be supported again.
