from typing import Generator, TypeVar, Generic, Tuple, List
import itertools
import time
import random

import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

V = TypeVar('V')


class GenCacher(Generic[V]):
    """
    Wrapper around a generator that stores the already sampled values and allows indexing.
    """

    def __init__(self, generator: Generator[V, None, None]):
        self._g = generator
        self._cache = []
        self._is_exhausted = False

    def is_exhausted(self) -> bool:
        return self._is_exhausted

    def get_cache_len(self) -> int:
        return len(self._cache)

    def fill_cache(self, idx: int):
        while not self.is_exhausted() and idx >= self.get_cache_len():
            # while len(self._cache) <= idx:
            try:
                self._cache.append(next(self._g))
            except StopIteration as _:
                self._is_exhausted = True

    def __getitem__(self, idx: int) -> V:
        self.fill_cache(idx)
        return self._cache[idx]


def summations(sum_to: int, ls: List[int]) -> Generator[Tuple, None, None]:
    if len(ls) == 1:
        if sum_to < ls[0]:
            yield sum_to,
    else:
        for head in range(min(sum_to + 1, ls[0])):
            for tail in summations(sum_to - head, ls[1:]):
                yield (head,) + tail


def product(gens: List[Generator]) -> Generator:
    """
    Utility to compute the cartesian product between an arbitrary number of generators.
    Developed to handle the case of a possible mix of finite and infinite generators.
    The built-in itertools.product can only compute the cartesian product between finite generators.

    The exploration strategy can be visualized using the following code block:

    .. code-block:: python
        :linenos:
        :emphasize-lines: 21

        import matplotlib.pyplot as plt
        import trainer.demo_data as dd
        import trainer.lib as lib

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim3d([0.0, 10.0])
        ax.set_xlabel('X')
        ax.set_ylim3d([0.0, 10.0])
        ax.set_ylabel('Y')
        ax.set_zlim3d([0.0, 10.0])
        ax.set_zlabel('Z')
        xs, ys, zs = [], [], []

        gens = [
            dd.finite_test_gen(start=0, end=3),
            dd.infinite_test_gen(first=0),
            dd.finite_test_gen(start=0, end=3)
        ]

        for c in lib.product(gens):
            xs.append(c[0])
            ys.append(c[1])
            zs.append(c[2])
            ax.plot(xs=xs[-2:], ys=ys[-2:], zs=zs[-2:])
            fig.show()
            plt.pause(0.01)

    The result looks as following:

    .. image:: ../media/gen_product_exploration.gif

    :param gens: Between 1 and N generators
    :return: One generator that returns all N-tuples, built from the input generators
    """
    gens = list(map(GenCacher, gens))

    for distance in itertools.count(0):
        changed = False
        for gen in gens:
            gen.fill_cache(distance)
        for idxs in summations(distance, [gen.get_cache_len() for gen in gens]):
            # print(idxs)
            res = tuple(gen[idx] for gen, idx in zip(gens, idxs))
            yield res
            changed = True
        if not changed:
            return


def sample_randomly(gens: List[Generator], probas: List[float]):
    while gens:
        i = np.random.choice(range(len(gens)), 1, p=softmax(probas))[0]
        try:
            x = next(gens[i])
            yield x
        except StopIteration as e:
            gens.pop(i)
            probas.pop(i)


if __name__ == '__main__':
    import trainer.demo_data as dd
    gens = [dd.finite_test_gen(start=0, end=5), dd.finite_test_gen(start=10, end=15)]

    for x in sample_randomly(gens, [0.5, 0.5]):
        print(x)
