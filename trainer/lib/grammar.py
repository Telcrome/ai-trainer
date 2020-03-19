from __future__ import annotations
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Tuple

import trainer.lib as lib


class Symbol:
    name = 'GenericSymbol'


class TS(Symbol):
    pass


class NTS(Symbol):
    pass


class NonTermNode(lib.TreeNode[NTS, lib.TreeNode]):  # lib.TreeNode as child is a workaround for making annotations work
    pass


class RuleNode(lib.TreeNode[List[Symbol], NonTermNode]):
    def asdf(self):
        self.value.


class Program:

    def __init__(self, start_symbol: type(NTS)):
        self.program: List[type(Symbol)] = [start_symbol]

    def get_nts(self) -> Union[type(NTS), None]:
        for i, sym in enumerate(self.program):
            if issubclass(sym, NTS):
                return i, sym
        return -1, None

    def use_rule(self, index: int, rule: List[Union[type(NTS), type(TS)]]):
        self.program.pop(index)
        for i, s in enumerate(rule):
            self.program.insert(index + i, s)


class Grammar:
    prod_rules: Dict[type(NTS), List[Tuple[List[Union[type(NTS), type(TS)]], int]]] = {}

    def __init__(self, start_symbol: type(NTS)):
        self.start_symbol: NTS = start_symbol

    def build_random_word(self) -> List[TS]:
        p = Program(self.start_symbol)
        p_index, nts = p.get_nts()
        while nts is not None:
            prod_rule, proba = random.choice(self.prod_rules[nts])

            p.use_rule(p_index, prod_rule)

            p_index, nts = p.get_nts()
        return p.program

    def __repr__(self):
        res = ""
        for prod_rule_key in self.prod_rules:
            right_repr = ''
            for rule in self.prod_rules[prod_rule_key]:
                prod_abbreviation = [s.name for s in rule[0]]
                right_repr += f" {prod_abbreviation} ({rule[1]}) | "
            res += f'{prod_rule_key.name} -> {right_repr[:-3]}'
        return res
