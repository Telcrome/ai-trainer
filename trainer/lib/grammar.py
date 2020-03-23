from __future__ import annotations
import itertools
import infinite
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Tuple, Callable, Iterable, Generator, TypeVar
from functools import reduce

import numpy as np
from scipy.special import softmax

import trainer.lib as lib


class Symbol:

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return self.name


class TS(Symbol):
    pass


class NTS(Symbol):
    pass


RULE = List[Tuple[List[Symbol], int]]


class Grammar:
    prod_rules: Dict[NTS, RULE] = {}

    def __init__(self, start_symbol: NTS):
        self.start_symbol: NTS = start_symbol

    def get_rule(self, nts: NTS) -> RULE:
        if nts in self.prod_rules:
            return self.prod_rules[nts]
        else:
            raise Exception(f"There is no rule for NTS {nts}")

    def __repr__(self):
        res = ""
        for prod_rule_key in self.prod_rules:
            right_repr = ''
            for rule in self.prod_rules[prod_rule_key]:
                prod_abbreviation = [s.name for s in rule[0]]
                right_repr += f" {prod_abbreviation} ({rule[1]}) | "
            res += f'{prod_rule_key} -> {right_repr[:-3]}\n'
        return res


class ProgramSearchTree:

    def __init__(self, grammar: Grammar):
        self.grammar = grammar
        self.max_depth = 10
        self.rule_counter: Dict[Symbol, int] = {key: 0 for key in self.grammar.prod_rules}

    def read_program(self) -> Iterable[Union[List[TS], None]]:
        for item in self._read_symbol(0, self.grammar.start_symbol):
            yield item

    def _read_symbol(self, depth: int, sym: Symbol) -> List[TS]:
        if isinstance(sym, TS):
            yield [sym]
        elif isinstance(sym, NTS):
            # if depth < self.max_depth:
            rules, probas = [], []
            for substitution, p in self.grammar.get_rule(sym):
                rules.append(substitution)
                probas.append(p)

            # p_arr = np.array(probas)
            # random_indices = np.random.choice(len(rules), len(rules), p=softmax(probas), replace=False)
            random_indices = np.arange(len(rules))
            # For making it probabilistic: Sort probabilistic
            # shuffle_list = zip(rules, probas)
            # random.shuffle(shuffle_list)
            # rules, probas = zip(*shuffle_list)

            for rule_i in random_indices:
                gens = [self._read_symbol(depth + 1, sym) for sym in rules[rule_i]]
                for rule_tuple in product(*gens):
                    res = reduce(lambda x, y: x + y, [i for i in rule_tuple])
                    yield res
            # else:
            #     yield []
        else:
            raise Exception(f"Cannot read symbol {sym}")
