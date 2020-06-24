from __future__ import annotations
import inspect
import functools
import typing
from enum import Enum
import itertools
import random
from abc import ABC, abstractmethod
from typing import get_type_hints, Optional, Dict, List, Union, Tuple, Callable, Generator, Any
from functools import reduce

import numpy as np


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
        For example, converts a function f: bool -> int to a generator f: Generator[bool] -> Generator[int]
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

    @staticmethod
    def is_callable(f: Any) -> bool:
        try:
            _ = f.__call__
            return True
        except AttributeError as _:
            return False

    def bind_object(self, o: Any):
        short_name = o.__qualname__.split('.')[-1]

        if isinstance(o, type(Enum)) or isinstance(o, list):
            o = DslSemantics.generate_enum(o)
        elif inspect.isgeneratorfunction(o):
            pass
        elif DslSemantics.is_callable(o):
            o = DslSemantics.gen_wrapper(o)
        else:
            raise Exception("This object cannot be binded, maybe remove parenthesis?")

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
