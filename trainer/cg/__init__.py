"""
Computational graphs package, contributes:

- A greedy, decision tree based, image transformation module
- A module for defining computational graphs using general python functions, trained using MCMC
"""
from trainer.cg.Dsl import Context, ProgPool
from trainer.cg.samplers import FloatSampler, EnumSampler, RandomNumber
