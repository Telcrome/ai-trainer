"""
Demonstrates how to generate programs of simple languages using the probabilistic grammar module.
We define an enum with non-terminals and substrings of the final word as terminal-symbols.
"""
import itertools
from typing import Any
from enum import Enum

from tqdm import tqdm

import trainer.lib as lib
import trainer.ml as ml


class NonTerminals(Enum):
    A, B = range(2)


if __name__ == '__main__':
    sg = lib.Grammar[Any, str](prod_rules={
        NonTerminals.A: [
            ([NonTerminals.A, NonTerminals.B], 1.),
            (['a'], 1.),
        ],
        NonTerminals.B: [
            (['b', NonTerminals.B], 1.),
            (['b'], 1.),
        ]
    }, ts_type=str, use_softmax=True)
    print(sg)

    for prog in tqdm(sg.read_program(NonTerminals.A)):
        print(prog)
