from __future__ import annotations
import os
from enum import Enum
from typing import TypeVar, NewType, Union, Callable, Tuple, Any, Dict, get_type_hints, List, Optional, Generic

import numpy as np
from graphviz import render

import trainer.lib as lib

Semantics = Union[Callable, Enum]


class CNodeType(Enum):
    FuncNode, ParamNode, EnumNode = range(3)


def val_to_label(v: Any, max_length=100, vis_depth=4) -> str:
    """
    Attempts to visualize any value using a string displayable in the dot language of graphviz

    :param v: The value to be visualized
    :param max_length: Maximum length of the resulting string
    :param vis_depth: maximum lines that the string may occupy
    :return:
    """
    if isinstance(v, tuple):
        res = '<br/>'.join([val_to_label(x) for x in v])
    elif isinstance(v, np.ndarray) and len(v.shape) > 1:
        res = f'{v.shape}, {v.dtype}, {np.unique(v, return_counts=True)}'[:max_length] + '<br/>'
        for i in range(min(vis_depth, v.shape[0])):
            res += f'{str(v[i, :])[1:-1]}<br align="left" />'
    else:
        res = str(v).replace('<', '').replace('>', '')

    return res


class CNode:
    @classmethod
    def from_json(cls, d: Dict, sem: Dict) -> CNode:
        d_key = list(d.keys())[0]
        res = cls(sem[d_key][0], sem[d_key][1])
        res.parents = [CNode.from_json(d[d_key][i], sem) for i, _ in enumerate(d[d_key])]
        return res

    def __init__(self, semantics: Semantics, node_type: CNodeType, name=''):
        self.id: Optional[int] = None
        # self.f_id: Optional[int] = None  # To identify the node across multiple features
        self.sem: Callable = semantics
        self.node_type: CNodeType = node_type
        if not name:
            self.name = semantics.__qualname__
        if node_type == CNodeType.FuncNode:
            self.arity = len(get_type_hints(semantics)) - 1
        else:
            self.arity = 0
        self.parents: List[CNode] = []
        self.last_res: str = ''

    def execute(self, store_last_result=False):
        """
        Executes the program using strict evaluation.
        """
        if self.node_type == CNodeType.ParamNode:
            # Just give the sampler the id if its a param node
            res = self.sem(self.id)
            if store_last_result:
                self.last_res = val_to_label(res)
            return res

        if self.arity == 0:
            res = self.sem()
            if store_last_result:
                self.last_res = val_to_label(res)
            return res

        params = [n.execute(store_last_result=store_last_result) for n in self.parents]

        res = self.sem(*params)
        if store_last_result:
            self.last_res = val_to_label(res)
        return res

    def get_dot(self, dot_id: int, p_id=-1) -> Tuple[str, int]:
        if self.node_type == CNodeType.ParamNode:
            color = '#37C8AE'
        elif self.node_type == CNodeType.FuncNode:
            color = '#399de5'
        else:
            raise Exception(f"Colour codes do not support type {self.node_type}")
        res = f'{dot_id} [label=<{self.name}, id: {self.id}<br/>{self.last_res}>, fillcolor="{color}"]'
        if p_id != -1:
            res += f'{dot_id} -> {p_id}[labeldistance = 2.5, headlabel = ""];'
        p_dot_id = dot_id
        for p_node in self.parents:
            p_dot_id += 1
            # edge_id = p_dot_id
            r_n, p_dot_id = p_node.get_dot(p_dot_id, p_id=dot_id)
            res += r_n
        return res, p_dot_id


class DslFunc:

    def __init__(self, root: CNode):
        # self.f_id = f_id
        self.root: CNode = root
        # root.f_id = f_id
        self.n_nodes = -1
        self._number_consecutively()

    def execute(self, store_result=False) -> Any:
        return self.root.execute(store_last_result=store_result)

    def _number_consecutively(self) -> None:
        start: List[CNode] = [self.root]
        id = 0
        while len(start) > 0:
            next = []
            for n in start:
                # if n.node_type == CNodeType.FuncNode:
                n.id = id
                # n.f_id = self.f_id
                id += 1
                next.extend(n.parents)
            start = next
        self.n_nodes = id

    def visualize(self, f_name='', dir_path='', delete_dot_after=True, instance_id=None) -> str:
        res = 'digraph Tree {'
        res += 'node [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;'
        res += 'edge [fontname=helvetica] ;'
        res += f'-1 [label=<{self.root.last_res}<br/>Instance ID:{instance_id}>, fillcolor="#e5833c"] ;'
        node_res, _ = self.root.get_dot(0)
        res += node_res
        res += f'0 -> -1[labeldistance = 2.5, labelangle = 45, headlabel = "Output"];'
        res += '}'

        if not dir_path:
            dir_path = lib.logger.get_absolute_run_folder()
        if not f_name:
            f_name = f'ProgTree_{len([x for x in os.listdir(dir_path) if x.endswith(".png")])}'
        f_path = os.path.join(dir_path, f_name)

        with open(f_path, 'w') as f:
            f.write(res)

        # Convert a .dot file to .png
        render('dot', 'png', f_path)

        if delete_dot_after:
            os.remove(f_path)

        return res
