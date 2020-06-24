"""
Computational graphs are a simple tool for dealing with program synthesis.

They are most commonly used by traditional, gradient descent based, deep learning frameworks.
In this example we define simple computational nodes for training a computational graph that classifies MNIST images.
"""
from timeit import default_timer as timer
from typing import TypeVar, NewType
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import trainer.lib as lib
import trainer.ml as ml
import trainer.cg as cg


######################################
# Start demonstration dsl definition #
######################################

class MathOperation(Enum):
    Addition, Multiplication = range(2)


class MathExecutionContext(cg.Context):
    def get_start(self) -> float:
        return self.state['start_value']


def random_number_wrapper(num: cg.RandomNumber) -> float:
    return num


def math_op(op: MathOperation, a: float, b: float) -> float:
    if op == MathOperation.Addition:
        return a + b
    else:
        return a * b


######################################
# End demonstration dsl definition #
######################################


if __name__ == '__main__':
    exec_context = MathExecutionContext()

    prog_pool: cg.ProgPool = cg.ProgPool(
        r_type=float,
        fs=[
            # float
            (math_op, 3.),
            (exec_context.get_start, 5.),
            (random_number_wrapper, 5.)],
        context=exec_context,
        samplers=[cg.EnumSampler(MathOperation)]
    )

    N_ARCHITECTURES = 1000
    N_GRAPHS = 100

    # Generate architectures
    prog_pool.sample_words(N_ARCHITECTURES)

    # Initialize N of the first generated architectures
    prog_pool.set_init_num(N_GRAPHS)
    prog_pool.initialize_instances()

    # a, b = prog_pool.compute_features([{'start_value': 0.5}], visualize_after=True, store=True)
    #
    # prog_pool.diffusion_move(temperature=1.)
    #
    # c, d = prog_pool.compute_features([{'start_value': 0.5}], visualize_after=True, store=True)

    # Performance test
    N_TESTS = 1000
    start = timer()
    for r in np.random.random(N_TESTS):
        res = prog_pool.compute_features([{'start_value': r}], visualize_after=False, store=False)
    seconds = timer() - start
    print(f"Executing your operation took me {seconds} seconds, that is {seconds/N_TESTS} seconds per loop")
