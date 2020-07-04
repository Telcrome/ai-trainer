import math
import random
from collections import namedtuple
from itertools import count
from PIL import Image

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import trainer.ml as ml

env = gym.make('CartPole-v0').unwrapped
actions = env.action_space

env.reset()
for _ in tqdm(range(1000)):
    env.render()
    done = env.step(env.action_space.sample())

    if done:
        break

env.close()
