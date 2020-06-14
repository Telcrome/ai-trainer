from __future__ import annotations
import os
import itertools
import json
import random
import functools
import copy

from enum import Enum
from abc import ABC
from typing import TypeVar, NewType, Union, Callable, Tuple, Any, Dict, get_type_hints, List, Optional, Generic

import trainer.lib as lib
from trainer.cg.DslFunc import DslFunc, CNodeType, CNode, Semantics
from trainer.cg.samplers import Sampler


class Context(ABC):
    """
    Can be used to hold the context dependent semantics, meant to be derived for a specific problem.
    For example for image classification it might define a function context.get_img(), returning the input image.
    """

    def __init__(self):
        self.state = {}

    def set_state(self, state: Dict):
        self.state = state


class Dsl:
    """Wrapper around grammar for generating computational graphs"""

    def __init__(self, c: Context, sample_dict: Dict, samplers: List[Sampler], sampler_weight=2.):
        self.context = c
        self.grammar = lib.Grammar[Any, str](prod_rules={}, ts_type=str, use_softmax=True)
        for s in samplers:
            if s.r_type not in self.grammar.prod_rules:
                self.grammar.prod_rules[s.r_type] = []
            self.grammar.prod_rules[s.r_type].append(
                (['{"' + s.name + '": []}'], sampler_weight)  # TODO allow user to tune this value
            )
        self.semantics: Dict[str, Tuple[Semantics, CNodeType]] = sample_dict

    def add_function(self, f: Callable, prio: float) -> None:
        """
        Add an arbitrary python function to your DSL.

        :param f: A well typed callable
        :param prio: The corresponding priority
        """
        self.grammar.append_semantics(f, prio)
        self.semantics[f.__qualname__] = (f, CNodeType.FuncNode)

    def sample_n_words(self, r_type: Any, max_n=10):
        res = [word for word in itertools.islice(self.grammar.sample_prog_strings(r_type), max_n)]
        return res


