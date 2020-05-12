from __future__ import annotations
import inspect
import json
import functools
import typing
from enum import Enum
import itertools
import infinite
import random
from abc import ABC, abstractmethod
from typing import get_type_hints, Optional, Dict, List, Union, Tuple, Callable, Generator, Any
from functools import reduce

import numpy as np

import trainer.lib as lib

NTS = typing.TypeVar('NTS')
TS = typing.TypeVar('TS')

RULE = List[Tuple[List[Union[TS, NTS]], float]]


def analyse_function_type(f: Callable) -> Tuple[List[type], type]:
    type_dict = get_type_hints(f)
    return [type_dict[key] for key in type_dict if key != 'return'], type_dict['return']


class Grammar(typing.Generic[NTS, TS]):

    def __init__(self, prod_rules: Dict[NTS, RULE], ts_type: Any, use_softmax=True):
        self.prod_rules: Dict[NTS, RULE] = prod_rules
        self.d = -1
        self.use_softmax = use_softmax
        self.ts_type = ts_type

    def append_semantics(self, f: Callable, prio: float):
        arg_types, r_type = analyse_function_type(f)
        if r_type not in self.prod_rules:
            self.prod_rules[r_type] = []

        rule_str = ['{"' + f.__qualname__ + '":[']
        for arg_type in arg_types:
            rule_str.append(arg_type)
            rule_str.append(',')
        rule_str.append(']}')

        new_rule = rule_str, prio
        self.prod_rules[r_type].append(new_rule)

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

    def sample_prog_strings(self, sym: NTS):
        for tss in self.read_program(sym):
            prog_str = ''.join([str(ts) for ts in tss[0]])
            yield prog_str, tss[1]

    def read_program(self, start_symbol: NTS) -> Generator[Union[List[TS], None]]:
        for item in self._read_symbol(0, start_symbol):
            yield item, self.d

    def _read_symbol(self, depth: int, sym: Union[NTS, TS]) -> Generator[List[TS]]:
        """
        Recursively create words from grammar

        :param depth: depth contains the current depth
        :param sym: Symbol that is expanded
        :return: Generator that iterates over all words that can be expanded from sym
        """
        if isinstance(sym, self.ts_type):
            self.d = depth  # Simple hack for outside functions to access depth of generated words
            yield [sym]
        else:
            rules, probas = [], []
            r = self.get_rule(sym)
            for substitution, p in r:
                rules.append(substitution)
                probas.append(p)

            rule_gens = []
            for rule in rules:
                sym_gens = [self._read_symbol(depth + 1, sym) for sym in rule]
                rule_gens.append(lib.product(sym_gens))

            for random_rule_gen in lib.sample_randomly(rule_gens, probas, use_softmax=self.use_softmax):
                yield reduce(lambda x, y: x + y, [i for i in random_rule_gen])


def prepend_gen(prep, gens):
    for item in gens:
        yield prep, [x for x in item]


def dsl_func(priority: float):
    def wrapper(func: Callable):
        return func, priority

    return wrapper
