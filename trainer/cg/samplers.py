import random
from enum import Enum
from abc import ABC, abstractmethod
from typing import TypeVar, NewType, Union, Callable, Tuple, Any, Dict, get_type_hints, List, Optional, Generic

import trainer.lib as lib

V = TypeVar('V')


class Sampler(ABC, Generic[V]):
    """
    Template for implementing nodes that can be sampled using MCMC
    """

    def __init__(self, name='MCSampler', r_type=V):
        self.name = name
        self.r_type = r_type
        self.delegate: Optional[Callable[[int], V]] = None

    @abstractmethod
    def resample(self, last_value: Optional[V] = None) -> V:
        raise NotImplementedError()

    def serialize(self, vals: List[V]) -> List[str]:
        return [str(v) for v in vals]

    @abstractmethod
    def deserialize(self, vals: List[str]) -> List[V]:
        raise NotImplementedError()

    def sample(self, node_id: int) -> V:
        return self.delegate(node_id)


RandomNumber = NewType('RandomNumber', float)


class SamplerFloat(Sampler[RandomNumber]):
    def __init__(self):
        super().__init__(name='RFloat', r_type=RandomNumber)

    def resample(self, last_value: Optional[V] = None) -> V:
        if last_value is None:
            return random.random()
        else:
            return (last_value + random.random()) / 2

    def deserialize(self, vals: List[str]) -> List[V]:
        return [float(v) for v in vals]


class SamplerEnum(Sampler[V]):

    def __init__(self, e: type(Enum)):
        self.e = e
        super().__init__(name=e.__qualname__, r_type=e)

    def resample(self, last_value: Optional[V] = None) -> V:
        choices = list(self.e)
        if last_value is None:
            return random.choice(choices)
        else:
            choices.remove(last_value)
            return random.choice(choices)

    def serialize(self, vals: List[V]) -> List[str]:
        return [v.value for v in vals]

    def deserialize(self, vals: List[str]) -> List[V]:
        c = lib.make_converter_dict_for_enum(self.e)
        res = [c[v] for v in vals]
        return res


class NumberSampler(Sampler[int]):

    def __init__(self, start: int, end: int, int_type: Any):
        self.t = int_type
        self.start = start
        self.end = end
        super().__init__(name='IntSampler', r_type=self.t)

    def resample(self, last_value: Optional[V] = None) -> V:
        return random.randint(self.start, self.end)

    def deserialize(self, vals: List[str]) -> List[V]:
        return [int(v) for v in vals]
