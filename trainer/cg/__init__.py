"""
Computational graphs package, contributes:

- A greedy, decision tree based, image transformation module
- A module for defining computational graphs using general python functions, trained using Simulated Annealing
"""
from trainer.cg.Dsl import Context
from trainer.cg.ProgPool import ProgPool
from trainer.cg.samplers import EnumSampler, RandomNumber, RandomInteger
