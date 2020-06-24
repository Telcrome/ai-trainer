"""
Given rules of a grammar the syn package searches for syntactic correct solutions
for example input/output pairs.
"""
from __future__ import annotations

from typing import List, Union, Set
from abc import ABC, abstractmethod

import numpy as np


class State:

    def __init__(self, pair: Pair):
        self.pair = pair
        # self.grid = np.zeros_like(pair.situation)
        self.grid = np.zeros((10, 1, 1), dtype=np.bool)
        self.grid[0] = True
        self.memory = False
        self.attention = np.zeros_like(self.grid[0, :, :], dtype=np.bool)
        # self.bg = np.zeros_like(grid[0])

    def visualize(self, ax1, ax2):
        raise NotImplementedError()

    def is_final(self) -> bool:
        """
        Can be used to determine for one

        :return True if the state is a valid solution to the output
        """
        raise NotImplementedError


class NonTerminalSym:

    def __init__(self, rule: SubstitutionRule):
        self.rule = rule

    def __getitem__(self, item: int):
        return self.rule.right_side[item]

    def __len__(self):
        return len(self.rule.right_side)


class TerminalSym:

    def apply_on_state(self, s: State):
        pass

    def semantics(self):
        pass


class SubstitutionRule:

    def __init__(self, left_side, right_side):
        self.left_side: NonTerminalSym = left_side
        self.right_side: List[List[Union[NonTerminalSym, TerminalSym]]] = right_side


if __name__ == '__main__':
    r1 = SubstitutionRule(
        'S',
        'FA'
    )
