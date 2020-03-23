from typing import Generator, TypeVar, Generic, Tuple, List
import itertools

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
        _ = self[0]

    def is_exhausted(self) -> bool:
        return self._is_exhausted

    def get_cache_len(self) -> int:
        return len(self._cache)

    def __getitem__(self, idx: int) -> V:
        while len(self._cache) <= idx:
            try:
                self._cache.append(next(self._g))
            except StopIteration as _:
                self._is_exhausted = True
        return self._cache[idx]


def get_combinations(level: int, lx: int, ly: int) -> Generator[Tuple, None, None]:
    for i in range(level):
        yield i, level
    if level <= lx and level <= ly:
        yield level, level
    for i in range(level):
        yield level, i


def product_two_gens(g1: GenCacher, g2: GenCacher) -> Generator:
    l1, l2 = 0, 0
    while not g1.is_exhausted() or not g2.is_exhausted():
        for coord in get_combinations(l1, l2):
            yield g1[coord[0]], g2[coord[1]]
        if not g1.is_exhausted():
            l1 += 1
        if not g2.is_exhausted():
            l2 += 1

    # l = 0
    # for level in itertools.count(0):
    #     level1 = min(level, g1.get_cache_len())
    #     level2 = min(level, g2.get_cache_len())
    #     for t in get_combinations(level1, level2):
    #         print(t)
    #         yield g1[t[0]], g2[t[1]]


def summations(sum_to: int, n=2) -> Generator[Tuple, None, None]:
    if n == 1:
        yield sum_to,
    else:
        for head in range(sum_to + 1):
            for tail in summations(sum_to - head, n - 1):
                yield (head,) + tail


def product(*gens):
    gens = list(map(GenCacher, gens))

    for distance in itertools.count(0):
        for idxs in get_combinations(gens):
            print(idxs)
            # res = tuple(gen[idx] for gen, idx in zip(gens, idxs))
            # yield res


if __name__ == '__main__':
    def finite_test_gen():
        for item in range(3):
            yield item


    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    fig, ax = plt.subplots()

    xs, ys, zs = [], [], []
    # for c in product_two_gens(GenCacher(finite_test_gen()), GenCacher(finite_test_gen())):
    #     xs.append(c[0])
    #     ys.append(c[1])
    #     # zs.append(c[2])
    #     ax.scatter(c[0], c[1], s=10, c=1)
    #     print(c)
    for l in range(10):
        for c in get_combinations(l, 0, 0):
            xs.append(c[0])
            ys.append(c[1])
            # zs.append(c[2])
            ax.scatter(c[0], c[1], s=l, c=l)
            ax.annotate(str(l), c)
            print(c)
            # ax.plot(xs=xs, ys=ys)
        # for t in summations(l, n=3):
        #     xs.append(t[0])
        #     ys.append(t[1])
        #     zs.append(t[2])
        #     ax.plot(xs=xs, ys=ys, zs=zs)
        #     print(t)
    ax.plot(xs, ys)
    fig.show()
