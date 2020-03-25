from typing import Generator, TypeVar, Generic, Tuple, List
import itertools
import time

import numpy as np
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


def product(*gens):
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
    i = 0  # TODO use probas and sample i
    while gens:
        try:
            x = next(gens[i])
            yield x
        except StopIteration as e:
            gens.pop(i)
            i = 0


if __name__ == '__main__':
    import trainer.demo_data as dd
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim3d([0.0, 10.0])
    ax.set_xlabel('X')

    ax.set_ylim3d([0.0, 10.0])
    ax.set_ylabel('Y')

    ax.set_zlim3d([0.0, 10.0])
    ax.set_zlabel('Z')

    xs, ys, zs = [], [], []
    for c in product(dd.finite_test_gen(), dd.infinite_test_gen(), dd.finite_test_gen()):
        # print(c)
        xs.append(c[0])
        ys.append(c[1])
        zs.append(c[2])
        # ax.scatter(c[0], c[1], s=10, c=1)
        # ax.plot(xs, ys)
        ax.plot(xs=xs, ys=ys, zs=zs)
        fig.show()
        plt.pause(0.5)
