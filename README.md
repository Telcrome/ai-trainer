# The Mission

<!-- What task does trainer solve? -->
There are still tasks that can be faster explained to a human than to a computer.
This project came to life for the purpose of solving that. 
Using trainer you can train artificial brains by stating a set of inputs and either your desired output or a combination of output and easily to define constraints.

<!-- Motivation -->
Take for example the following traditional medical multi-class classification problem:
Given one or more ultrasound videos, an xray and a textual description of the patient's symptoms,
predict the medical decision of care.
From an engineering perspective, this is a difficult task.
With trainer, tasks that involve decisions given numerous different inputs,
should be greatly simplified.

<!-- Dataset Format -->
Furthermore, trainer defines a convenient, subject-centered dataset format.
In the medical case, a subject can be a patient.
A patient can have multiple different studies
(Simple attributes, CT 3D data, ultrasound videos, wholeslide images),
which should all be usable by a trained AI.

# Installation for User

Open anaconda powershell, activate an environment with anaconda, navigate into the trainer repo and execute the following to install trainer using pip, including its dependencies:

```bash
pip install ai-trainer
```

For Online Learning you have to install PyTorch:
```bash
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

AI-Trainer helps with building a data generator and it relies on imgaug for it:

```bash
conda install imgaug -c conda-forge
```

# Getting started with training models

Trainer currently supports annotating images and videos.
First, create a dataset using

```bash
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

```bash
git clone https://github.com/Telcrome/ai-trainer
```

Both vsc and pycharm are used for development with
their configurations provided in ```.vscode``` and ```.idea```

## Recommended environments

For development we recommend to install the conda environment into a subfolder of the repo.
This allows for easier experimentation and the IDE expects it this way.

```bash
conda env create --prefix ./envs -f environment.yml
conda activate .\envs\.
```

Now install a deep learning backend.
PyTorch provides well-working [conda install commands](https://pytorch.org/get-started/locally/).

For Tensorflow with GPU:
```bash
conda install cudatoolkit=10.0 cudnn=7.6.0=cuda10.0_0
pip install tensorflow-gpu
```

### Testing Development for pip and cli tools

Installing the folder directly using pip does not work due to the large amount of files inside the local development folder,
especially because in the local development setup the environment is expected to be a subfolder of the repo.
```bash
pip install -e .
```

# Using Docker

Docker and the provided DOCKERFILE support is currently experimental as it proved to slow down the annotation GUI too much.
When the transition to a web GUI is completed docker will be supported again.