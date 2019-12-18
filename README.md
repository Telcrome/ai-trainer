# The Mission

TODO

# Installation for User

Open anaconda powershell, navigate into the trainer repo and execute:

```bash
pip install git+https://github.com/Telcrome/ai-trainer@master
```

For dependencies please see the file ```environment.yml```.

# Getting started

After activating the environment containing the trainer and its dependencies,
feel free to inspect some of the tutorials in ```./tutorials/```.

## Optional dependencies

Annotator helps with building a data generator and it relies on imgaug for it
```bash
conda config --add channels conda-forge
conda install imgaug
```

## Development Setup

Both vsc and pycharm are used for development with
their configurations provided in ```.vscode``` and ```.idea```

### Installing environments

For development we recommend to install the environment in a local folder.
This allows for easier experimentation and the IDE expects it this way.

```bash
conda env create --prefix ./envs -f environment.yml
conda activate .\envs\.
```

# Using Docker

Docker and the provided DOCKERFILE support is currently experimental as it proved to slow down the annotation GUI too much.
When the transition to a web GUI is completed docker will be supported again.