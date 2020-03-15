import os
import pathlib
import json

trainer_config_folder = os.path.join(str(pathlib.Path.home()), '.trainer')
BIG_BIN_DIRNAME = 'bin_bin'
BIG_BIN_KEY = 'big_bin_dir'
DB_CON_KEY = 'db_con'
config_path = os.path.join(trainer_config_folder, 'trainer_config.json')


def save_config_json(obj=None):
    if obj is None:
        obj = {
            BIG_BIN_KEY: os.path.join(trainer_config_folder, BIG_BIN_DIRNAME),
        }

    if not os.path.exists(obj[BIG_BIN_KEY]):
        os.mkdir(obj[BIG_BIN_KEY])

    with open(config_path, 'w') as f:
        json.dump(obj, f)
    return obj


def load_config_json():
    with open(config_path, 'r') as f:
        return json.load(f)


if not os.path.exists(trainer_config_folder):
    os.mkdir(trainer_config_folder)

if not os.path.exists(config_path):
    config = save_config_json()
else:
    config = load_config_json()
