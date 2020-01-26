import os
import multiprocessing as mp
import concurrent.futures
import random
from typing import Tuple, Iterable

import numpy as np
from tqdm import tqdm

from trainer.ml import Subject, Dataset
from trainer.ml.data_loading import random_subject_generator, get_mask_for_frame

NCORE = 4


def subject_processor(s: Subject) -> Tuple[np.ndarray, np.ndarray]:
    is_names = s.get_image_stack_keys()
    is_name = random.choice(is_names)
    available_structures = s.get_structure_list(image_stack_key=is_name)
    selected_struct = random.choice(list(available_structures.keys()))
    im = s.get_binary(is_name)
    selected_frame = random.randint(0, im.shape[0])
    return im, get_mask_for_frame(s, is_name, selected_struct, selected_frame)


def process_logic(job_queue: mp.Queue, res_queue: mp.Queue):
    job = job_queue.get()
    s = Subject.from_disk(job)
    res_queue.put(subject_processor(s))


def single_process(d: Dataset):
    ss = []
    jobs = [os.path.join(d.get_working_directory(), s_name) for s_name in d.get_subject_name_list()]
    for job in tqdm(jobs):
        ss.append(process_logic(job))
    return ss


def multi_process(d: Dataset):
    ss = []
    jobs = [os.path.join(d.get_working_directory(), s_name) for s_name in d.get_subject_name_list()]
    job_queue = mp.Queue()
    for job in jobs:
        job_queue.put(job)
    res_queue = mp.Queue()
    pool = mp.Pool(NCORE, initializer=process_logic, initargs=(job_queue, res_queue))
    n = 0
    while True:
        item = res_queue.get()
        n += 1
        print(f"{n}Append")
        ss.append(item)
        if len(ss) == len(jobs):
            break
    pool.close()
    pool.join()
    return ss


if __name__ == '__main__':
    ds = Dataset.download(url='https://rwth-aachen.sciebo.de/s/1qO95mdEjhoUBMf/download',
                          local_path='./data',  # Your local data folder
                          dataset_name='crucial_ligament_diagnosis'  # Name of the dataset
                          )
    # train_pairs = multi_process(ds)
