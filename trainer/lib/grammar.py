from __future__ import annotations
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Tuple
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

    lib.TreeNode as child type is a workaround for making annotations work, it should be RuleNode
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
        self.tree_root = SymbolNode(self.grammar.start_symbol)
        self.expand_nodes: List[SymbolNode] = [self.tree_root]  # Store the nodes that can still be expanded

    def expand_node(self, node: SymbolNode) -> None:
        # node = self.expand_nodes.pop(0)
        for substitution, p in self.grammar.get_rule(node.value):
            exp_node = ExpressionNode(substitution, parent=node)
            node.children.append(exp_node)

            for sym in exp_node.value:
                sym_node = SymbolNode(sym, parent=exp_node)
                if isinstance(sym, NTS):
                    self.expand_nodes.append(sym_node)
                exp_node.children.append(sym_node)

    def read_program(self, node: SymbolNode) -> List[TS]:
        if isinstance(node.value, TS):
            return [node.value]
        elif isinstance(node.value, NTS):
            if not node.children:
                self.expand_node(node)

            # Choose an expression to go further from here
            exp_node = random.choice(node.children)

            progs = [self.read_program(sym_node) for sym_node in exp_node.children]
            res = []
            for prog in progs:
                for a in prog:
                    res.append(a)
            # res = reduce(lambda p1, p2: p1.extend(p2), progs)
            return res
        else:
            raise Exception(f"Can only read nodes that contain single symbols, not {node.value}")
