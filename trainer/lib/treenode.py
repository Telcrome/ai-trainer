"""
Defines utility functions for working with tree structures
"""
from __future__ import annotations
from typing import TypeVar, Generic, List, Union

V = TypeVar('V')
C = TypeVar('C')


class TreeNode(Generic[V, C]):
    """
    Represents an ordered tree with
    """

    def __init__(self, value: V, parent: Union[C, None] = None):
        self.children: List[TreeNode] = []
        self.value = value
