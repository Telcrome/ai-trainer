from __future__ import annotations
import itertools
import infinite
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Tuple, Callable, Iterable
from functools import reduce

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


class SymbolNode(lib.TreeNode[Symbol, lib.TreeNode]):
    """
    Node of ProgramSearchTree

    lib.TreeNode as child type is a workaround for making annotations work, it should be ExpressionNode
    """
    pass


class ExpressionNode(lib.TreeNode[List[Symbol], SymbolNode]):
    pass


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

    # def read_program(self) -> Iterable[Union[List[TS], None]]:
    #     for item in self._read_symbol(self.grammar.start_symbol):
    #         yield item

    def _read_symbol(self, sym: Symbol):
        if isinstance(sym, TS):
            yield [sym]
        elif isinstance(sym, NTS):
            rules, probas = [], []
            for substitution, p in self.grammar.get_rule(sym):
                rules.append(substitution)
                probas.append(p)

            # For making it probabilistic: Sort probabilistic
            for rule in rules:
                gens = [self._read_symbol(sym) for sym in rule]
                for rule_tuple in itertools.product(*gens):
                    res = reduce(lambda x, y: x + y, [i for i in rule_tuple])
                    yield res
            # for exp_node in sym_node.get_all_children():
            #     res = [prog for prog in self._read_expression(exp_node)]
            #     yield res
        else:
            raise Exception(f"Can only read nodes that contain single symbols, not {sym_node.value}")
