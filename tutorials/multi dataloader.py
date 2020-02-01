import time
import os
import random
from typing import Tuple

from tqdm import tqdm
import numpy as np
import cv2
from torch.multiprocessing import Process, Queue, Lock

import trainer.ml as ml
from trainer.ml.data_loading import random_subject_generator, get_mask_for_frame


def producer(queue, lock, ds_path: str):
    # Synchronize access to the console
    with lock:
        print('Starting producer => {}'.format(os.getpid()))

    ds = ml.Dataset.from_disk(ds_path)

    def preprocess(s: ml.Subject) -> Tuple[np.ndarray, np.ndarray]:
        is_names = s.get_image_stack_keys()
        is_name = random.choice(is_names)
        available_structures = s.get_structure_list(image_stack_key=is_name)
        selected_struct = random.choice(list(available_structures.keys()))
        im = s.get_binary(is_name)
        possible_frames = s.get_masks_of(is_name, frame_numbers=True)
        selected_frame = random.choice(possible_frames)
        gt = get_mask_for_frame(s, is_name, selected_struct, selected_frame)

        # Processing
        im = cv2.resize(im[selected_frame], (384, 384))
        im = np.rollaxis(ml.normalize_im(cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)), 2, 0)
        gt = gt.astype(np.float32)
        gt = cv2.resize(gt, (384, 384))
        # gt = np.expand_dims(gt, 0)
        gt_stacked = np.zeros((2, gt.shape[0], gt.shape[1]), dtype=np.float32)
        gt_stacked[0, :, :] = gt.astype(np.float32)
        gt_stacked[1, :, :] = np.invert(gt.astype(np.bool)).astype(gt_stacked.dtype)

        return im, gt_stacked

    g = random_subject_generator(ds, preprocess, split='train', batchsize=4)

    # Place our names on the Queue
    for te in g:
        queue.put(te)

    # Synchronize access to the console
    with lock:
        print('Producer {} exiting...'.format(os.getpid()))


def consumer(queue, lock):
    # Synchronize access to the console
    with lock:
        print('Starting consumer => {}'.format(os.getpid()))

    lw = ml.LogWriter()

    # Train
    for _ in tqdm(range(10000)):
        # If the queue is empty, queue.get() will block until the queue has data
        x, y = queue.get()
        lw.save_tensor(x, name='input_tensor')
        del x
        # Synchronize access to the console
    with lock:
        print(f'{os.getpid()} finished')


if __name__ == '__main__':

    # Create the Queue object
    queue = Queue()

    # Create a lock object to synchronize resource access
    lock = Lock()

    producers = [Process(target=producer, args=(queue, lock, './data/full_ultrasound_seg_0_0_9')) for _ in range(4)]
    consumers = [
        Process(target=consumer, args=(queue, lock))
    ]

    # Create consumer processes
    # for i in range(len(names) * 2):
    #     p = Process(target=consumer, args=(queue, lock))
    #
    #     # This is critical! The consumer function has an infinite loop
    #     # Which means it will never exit unless we set daemon to true
    #     p.daemon = True
    #     consumers.append(p)

    # Start the producers and consumer
    # The Python VM will launch new independent processes for each Process object
    for p in producers:
        p.start()

    for c in consumers:
        c.start()

    # Like threading, we have a join() method that synchronizes our program
    for p in producers:
        p.join()

    print('Parent process exiting...')

# import torch.multiprocessing as mp
# from tqdm import tqdm
#
# import trainer.ml as ml
#
# if __name__ == '__main__':
#     ds = ml.Dataset.from_disk('./data/full_ultrasound_seg_0_0_9')
#
#     structure_name = 'gt'  # The structure that we now train for
#     loss_weights = (0.1, 1.5)
#     BATCH_SIZE = 4
#     EPOCHS = 60
#
#     N = ds.get_subject_count(split='train')
#
#     lw = ml.LogWriter()
#
#     model = ml.seg_network.SegNetwork("ResNet_UNet", 3, 2, ds, batch_size=BATCH_SIZE)
#     for i in tqdm(range(1000)):
#         x, y = model.sample_minibatch()
#         lw.save_tensor(x, name='input_tensor')
