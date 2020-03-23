from typing import Generator, TypeVar, Generic, Tuple
import itertools

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

V = TypeVar('V')


class GenCacher(Generic[V]):
    """
    Wrapper around a generator that stores the already sampled values and allows indexing.
    """

    def __init__(self, generator: Generator[V, None, None]):
        self._g = generator
        self._cache = []
        self._is_finite = False

    def is_finite(self) -> bool:
        return self._is_finite

    def get_cache_len(self) -> int:
        return len(self._cache)

    def __getitem__(self, idx: int) -> V:
        while len(self._cache) - 1 <= idx:
            try:
                self._cache.append(next(self._g))
            except StopIteration as _:
                self._is_finite = True
        return self._cache[idx]


def get_combinations(level1: int, level2: int) -> Generator[V, None, None]:
    for t in zip(range(level1 + 1), reversed(range(level2 + 1))):
        yield t


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
        for idxs in summations(distance, len(gens)):
            res = tuple(gen[idx] for gen, idx in zip(gens, idxs))
            yield res


if __name__ == '__main__':
    def finite_test_gen():
        for item in range(3):
            yield item

    xs, ys = [], []
    for l in range(100):
        for c in get_combinations(min(l, 3), l):
            xs.append(c[0])
            ys.append(c[1])
            plt.scatter(c[0], c[1], s=l, c=l)
            print(c)
    plt.plot(xs, ys)
    plt.show()
