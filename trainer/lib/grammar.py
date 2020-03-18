from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Union


class TS:
    pass


class NTS:
    pass


class Grammar:

    def __init__(self, start_symbol: type(NTS)):
        self.start_symbol: NTS = start_symbol
        self.prod_rules: Dict[type(NTS), List[List[Union[type(NTS), type(TS)]]]] = {}

    def add_prod_rule(self, left: type(NTS), right: List[List[Union[type(NTS), type(TS)]]]):
        self.prod_rules[left] = right

    def __repr__(self):
        res = ""
        for prod_rule_key in self.prod_rules:
            right_repr = ''
            for rule in self.prod_rules[prod_rule_key]:
                right_repr += f"{rule} | "
            res += f'{prod_rule_key} -> {right_repr}'
        return res
