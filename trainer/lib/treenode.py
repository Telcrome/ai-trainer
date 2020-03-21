"""
Defines utility functions for working with tree structures
"""
from __future__ import annotations
from typing import TypeVar, Generic, List, Union

V = TypeVar('V')
C = TypeVar('C')


class TreeNode(Generic[V, C]):
    """
    Represents an ordered tree with different types of nodes.

    Every node just has to specify a value type and a children/parent type.
    """

    def __init__(self, value: V, parent: Union[C, None] = None):
        self._children: List[TreeNode] = []
        self.value = value
        self.parent = parent

    def add_child(self, c: C):
        self._children.append(c)

    def get_all_children(self):
        return self._children

    def __repr__(self):
        return f"Node {type(self)}\nValue: {self.value}\nChildren:{[type(c) for c in self._children]}"
