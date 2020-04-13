from __future__ import annotations
import inspect
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

    def read_program(self, start_symbol) -> Generator[Union[List[str], None]]:
        for item in self._read_symbol(0, start_symbol):
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
            r = self.get_rule(sym)
            # print(r)
            for substitution, p in r:
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


def prepend_gen(prep, gens):
    for item in gens:
        yield prep, [x for x in item]


def dsl_func(priority: float):
    def wrapper(func: Callable):
        return func, priority

    return wrapper


class DslSemantics(ABC):
    """
    Defines execution context and utility functionality for running python statements from strings.

    Child classes define methods that can be called inside of the DSL constructs.
    """

    def __init__(self, max_resources=10000):
        self.resources, self.max_resources, self.state = 0, max_resources, {}
        self.fs: Dict[str, Any] = {}
        self.prog, self.prog_str = None, ''

    def compile_prog(self, prog: str):
        self.prog_str = prog
        self.prog = compile(prog, 'dslprog', mode='eval')

    @staticmethod
    def generate_enum(e: type(Enum)) -> Generator:
        for v in e:
            yield v.value, str(e)

    @staticmethod
    def gen_wrapper(f: Callable) -> type(Generator):
        """
        Converts a function f: bool -> int to a generator f: Generator[bool] -> Generator[int]
        :param f: A callable
        :return: A generator with the semantics of f
        """
        def gen_f(*iters) -> Generator:
            if iters:
                for t in itertools.product(*iters):
                    ps = [p for p, _ in t]
                    ss = reduce(lambda s1, s2: f'{s1}, {s2}', [s for _, s in t])
                    yield f(*ps), ss
            else:
                yield f(), f.__qualname__.split('.')[-1]

        return gen_f

    # noinspection PyBroadException,PyStatementEffect
    @staticmethod
    def is_callable(f: Any) -> bool:
        try:
            f.__call__
            return True
        except:
            return False

    def bind_object(self, o: Any):
        short_name = o.__qualname__.split('.')[-1]

        if isinstance(o, type(Enum)) or isinstance(o, list):
            o = DslSemantics.generate_enum(o)
        elif inspect.isgeneratorfunction(o):
            pass
        elif DslSemantics.is_callable(o):
            o = DslSemantics.gen_wrapper(o)

        self.fs[short_name] = o

    def execute_program(self, state: Dict) -> Generator:
        self.state = state
        self.resources = self.max_resources
        try:
            res = eval(
                self.prog,
                self.fs,
                self.state
            )
        except Exception as e:
            print(f'Tried to execute {self.prog}')
            print(e)
            res = None
        return res
