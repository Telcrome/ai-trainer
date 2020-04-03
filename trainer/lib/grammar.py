from __future__ import annotations
import functools
import typing
from enum import Enum
import itertools
import infinite
import random
from abc import ABC, abstractmethod
from typing import get_type_hints, Optional, Dict, List, Union, Tuple, Callable, Generator
from functools import reduce

import numpy as np

import trainer.lib as lib


def get_func_name(f: Callable) -> str:
    return f.__qualname__


class SyntaxNode:

    def __init__(self, f: Callable, score: float):
        self.f = f
        self.score = score
        type_dict = typing.get_type_hints(self.f)
        self.return_type: type = type_dict['return']
        self.child_types: List[type] = [type_dict[key] for key in type_dict if key != 'return']
        print(type_dict)
        self._is_terminal = len(self.child_types) > 0

    def is_terminal(self) -> bool:
        return self._is_terminal


NTS = typing.TypeVar('NTS')
RULE = List[Tuple[List[Union[str, NTS]], float]]


class Grammar(typing.Generic[NTS]):

    def __init__(self, prod_rules: Dict[NTS, RULE], start_symbol: Enum):
        self.prod_rules: Dict[NTS, RULE] = prod_rules
        self.start_symbol = start_symbol

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

    def sample_prog_strings(self):
        for tss in self.read_program():
            prog_str = ''.join([str(ts) for ts in tss])
            # prog_str = reduce(lambda x, y: str(x) + str(y), tss)
            yield prog_str

    def read_program(self) -> Generator[Union[List[TS], None]]:
        for item in self._read_symbol(0, self.start_symbol):
            yield item

    def _read_symbol(self, depth: int, sym: Union[NTS, str]) -> Generator[List[str]]:
        """

        :param depth:
        :param sym:
        :return:
        """
        if isinstance(sym, str):
            yield [sym]
        else:
            rules, probas = [], []
            for substitution, p in self.get_rule(sym):
                rules.append(substitution)
                probas.append(p)

            rule_gens = []
            for rule in rules:
                sym_gens = [self._read_symbol(depth + 1, sym) for sym in rule]
                rule_gens.append(lib.product(sym_gens))

            for random_rule_gen in lib.sample_randomly(rule_gens, probas):
                yield reduce(lambda x, y: x + y, [i for i in random_rule_gen])


def analyse_function_type(f: Callable) -> Tuple[List[type], type]:
    type_dict = get_type_hints(f)
    return [type_dict[key] for key in type_dict if key != 'return'], type_dict['return']


def f_to_str(f: Callable):
    args, ret = analyse_function_type(f)
    params = " ".join([str(args)])
    return f'{get_func_name(f)} {params}'


def prepend_gen(prep, gens):
    for item in gens:
        yield prep, [x for x in item]


def dsl_func(priority: float):
    def wrapper(func: Callable):
        return func, priority

    return wrapper


class DomainLang(ABC):

    def __init__(self, start_symbol: type, max_resources=10000):
        self.start_symbol = start_symbol
        self.state = {}
        self.fs: List[Tuple[Callable, float]] = []
        self.constants: List[Enum] = []
        self.max_resources = max_resources
        self.resources = 0

    def set_state(self, state):
        self.state = state

    def execute_program(self, prog: Tuple[Callable, List], state: Dict):
        self.resources = self.max_resources
        self._execute_program(prog)

    def _execute_program(self, p: Tuple[Callable, List]) -> Optional[type]:
        self.resources -= 1
        if self.resources <= 0:
            return None
        params = p[1]
        if not params:
            return p[0]()
        parameters = [self._execute_program(param) for param in params]
        if None in parameters:
            return None
        return p[0](*parameters)

    def get_prog_repr(self, p: Tuple[Callable, List], depth=1) -> str:
        if p[1]:
            tabs = "  ".join(['' for _ in range(depth + 1)])
            param_repr = f'\n{tabs}'.join([''] + [self.get_prog_repr(param, depth + 1) for param in p[1]])
        else:
            param_repr = ''
        return f'{get_func_name(p[0])}{param_repr}'


class Dsl:
    def __init__(self, lang: DomainLang):
        self.lang: DomainLang = lang
        self.prod_rules: Dict[type, List[Tuple[Callable, float]]] = {}

        for f, score in lang.fs:
            # args, ret = analyse_function_type(f)
            self.add_rule(f, score)

            # args, ret = analyse_function_type(f)
            # fulltype = Callable[[*args], ret]
            #
            # def f_wrapper() -> fulltype:
            #     return f
            #
            # self.add_rule(f_wrapper, score)
        for c in lang.constants:
            for c_val in c:
                def func_wrapper() -> c:
                    return c_val

                self.add_rule(func_wrapper, 1.)

    def add_rule(self, f: Callable, score: float):
        args, ret = analyse_function_type(f)
        # fulltype = Callable[[*args], ret]

        if ret not in self.prod_rules:
            self.prod_rules[ret] = []
        # if fulltype not in self.prod_rules:
        #     self.prod_rules[fulltype] = []

        self.prod_rules[ret].append((f, score))
        # self.prod_rules[fulltype].append((f, score))

    def __repr__(self):
        res = ""
        for prod_rule_key in self.prod_rules:
            right_repr = ''
            for substitution in self.prod_rules[prod_rule_key]:
                prod_abbreviation = f_to_str(substitution[0])
                right_repr += f" {prod_abbreviation} ({substitution[1]}) | "
            res += f'{prod_rule_key} -> {right_repr[:-3]}\n'
        return res

    def read_program(self) -> Generator:
        for item in self._read_symbol(0, self.lang.start_symbol):
            yield item

    def _read_symbol(self, depth: int, sym: type) -> Generator:
        rules, probas = [], []
        for substitution, p in self.prod_rules[sym]:
            rules.append(substitution)
            probas.append(p)

        rule_gens = []
        for rule in rules:
            param_syms = analyse_function_type(rule)[0]
            if not param_syms:
                rule_gens.append((rule, []))
            else:
                sym_gens = [self._read_symbol(depth + 1, param_sym) for param_sym in param_syms]
                g = prepend_gen(rule, lib.product(sym_gens))
                rule_gens.append(g)

        for random_rule_gen in lib.sample_randomly(rule_gens, probas):
            # yield reduce(lambda x, y: x + y, [i for i in random_rule_gen])

            yield random_rule_gen
