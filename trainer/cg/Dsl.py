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

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import trainer.lib as lib
from trainer.demo_data.arc import Value
from trainer.cg.DslFunc import DslFunc, CNodeType, CNode, Semantics
from trainer.cg.samplers import Sampler


class Context(ABC):
    """Can be used to hold the context dependent semantics"""

    def __init__(self):
        self.state = {}

    def set_state(self, state: Dict):
        self.state = state


class Dsl:
    """Wrapper around grammar for generating program trees"""

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

    def add_function(self, f: Callable, prio: float):
        self.grammar.append_semantics(f, prio)
        self.semantics[f.__qualname__] = (f, CNodeType.FuncNode)

    def sample_n_words(self, r_type: Any, max_n=10):
        res = [word for word in itertools.islice(self.grammar.sample_prog_strings(r_type), max_n)]
        return res


class ProgInstance:

    @classmethod
    def from_json(cls, state: Dict, s_dict: Dict[str, Sampler]):
        res = cls(-1)
        res.state = state['state']
        node_states = {}
        for node_id in state['node_states']:
            vals, sampler_name = state['node_states'][node_id]
            sampler = s_dict[sampler_name]
            node_states[node_id] = sampler.deserialize(vals), sampler
        res.node_state = node_states
        res.stats = state['stats']
        # res.state['instance_id'] = new_id
        return res

    def __init__(self, architecture_id: int):
        self.state = {
            'instance_id': -1,
            'architecture_id': architecture_id,
            'locked': 0
        }
        self.node_state: Dict[int, Tuple[List[Any], Sampler]] = {}
        self.stats = {
            'used': 0
        }

    def duplicate(self) -> ProgInstance:
        copy_dict = copy.deepcopy(self.state)
        new_instance = ProgInstance(self.get_architecture_id())
        new_instance.state = copy_dict
        return new_instance

    def get_json(self):
        node_states = {}
        for node_id in self.node_state:
            vals, s = self.node_state[node_id]
            node_states[node_id] = s.serialize(vals), s.name
        return {
            'state': self.state,
            'node_states': node_states,
            'stats': self.stats
        }

    def get_instance_id(self) -> int:
        return self.state['instance_id']

    def set_instance_id(self, new_id: int) -> None:
        self.state['instance_id'] = new_id

    def get_architecture_id(self) -> int:
        return self.state['architecture_id']

    def lock(self):
        self.state['locked'] += 1

    def unlock(self):
        self.state['locked'] -= 1 if self.state['locked'] > 0 else 0

    def is_locked(self):
        return self.state['locked'] > 0

    def get_node_dict(self, n_id: int):
        return self.node_state[n_id] if n_id in self.node_state else None

    def resample_node(self, n_id: int) -> Any:
        sampler = self.node_state[n_id][1]
        new_value = sampler.resample(last_value=self.node_state[n_id][0][-1])
        self.node_state[n_id][0].append(new_value)
        return new_value

    def revert_resampling(self, n_id):
        self.node_state[n_id][0].pop()

    def get_value(self, n_id: int):
        if n_id not in self.node_state:
            return None
        return self.node_state[n_id][0][-1]

    def set_value(self, n_id, new_val: Any, s: Sampler) -> Any:
        if n_id not in self.node_state:
            self.node_state[n_id] = ([], s)
        self.node_state[n_id][0].append(new_val)
        return new_val

    def pick_node_id(self) -> Optional[int]:
        if self.node_state:
            return random.choice(list(self.node_state))
        else:
            return None

    def increment_used(self):
        self.stats['used'] += 1


class ProgPool:

    @classmethod
    def from_disk(cls, p: str, r_type: Any, fs: List[Tuple[Callable, float]], c: Context, samplers: List[Sampler]):
        res = cls(r_type, fs, c, samplers)

        # Load from disk
        df_words = pd.read_excel(p, index_col=0)
        json_path = f'{os.path.splitext(p)[0]}.json'
        with open(json_path, 'r') as f:
            instances_json = json.load(f)

        words_ls = [(word, depth) for i, (word, depth) in df_words.iterrows()]
        res.words = words_ls
        res._parse_progs()

        s_dict = {s.name: s for s in samplers}

        for inst_key in instances_json:
            res.instances[inst_key] = ProgInstance.from_json(instances_json[inst_key], s_dict)

        return res

    def __init__(self, r_type: Any, fs: List[Tuple[Callable, float]], context: Context, samplers: List[Sampler],
                 max_nodes=25):
        start_dict = {s.name: (s.sample, CNodeType.ParamNode) for s in samplers}
        self._dsl = Dsl(context, start_dict, samplers)
        for f, prio in fs:
            self._dsl.add_function(f, prio)
        self.r_type = r_type
        self.dsl_fs: List[DslFunc] = []
        self.words: List[Tuple[str, int]] = []  # [(string repr, grammar depth)]
        self.samplers = []
        self.instances: Dict[int, ProgInstance] = {}
        self.last_step: List[Tuple[int, int]] = []
        self.last_instance_invoked: int = -1
        self.new_instance_enumerator = 0
        self.set_samplers(samplers)
        self.last_killed_instances: Optional[List[ProgInstance]] = None
        self.last_birthed_instances: Optional[List[int]] = None
        self.init_n: int = 1
        self.max_nodes = max_nodes

    def set_init_num(self, n: int):
        self.init_n: int = n

    def to_disk(self, identifier: str, parent_dir='') -> str:
        file_path = os.path.join(parent_dir, f'{lib.create_identifier(identifier)}')

        # Save programs
        writer = pd.ExcelWriter(f'{file_path}.xlsx')
        df_words = pd.DataFrame(self.words, columns=('ProgString', 'Depth'))
        df_words.to_excel(writer, sheet_name='main')
        writer.save()

        # Save instantiations
        save_json = {}
        for inst_key in self.instances:
            save_json[inst_key] = self.instances[inst_key].get_json()
        with open(f'{file_path}.json', 'w') as f:
            json.dump(save_json, f)
        return file_path

    def get_locks(self):
        return len([True for key in self.instances if self.instances[key].is_locked()])

    def mcmc_step(self, temperature: float, dim_factor=3.5):
        """
        Resample one node of every program (which is not locked) inside the current instantiations.
        Performs birth and death moves randomly.
        """
        assert 0. <= temperature <= 1.

        instance_keys = np.array(list(self.instances.keys()))
        np.random.shuffle(instance_keys)

        # Death Moves
        desired_deaths = len(self.instances) - self.init_n if self.init_n < len(self.instances) else 1.
        num_deaths = min(np.floor((dim_factor * np.random.random()) * (desired_deaths * temperature)), len(self.instances))
        death_ids = np.random.choice(instance_keys, size=int(num_deaths), replace=False)
        self.last_killed_instances = [self.instances[death_id] for death_id in death_ids]
        for death_id in death_ids:
            self.remove_instance(death_id)

        # The number of diffusion steps, proportional to temperature
        num_steps = int(np.ceil(len(self.instances) * temperature))
        instance_keys = list(instance_keys)
        for death_id in death_ids:
            instance_keys.remove(death_id)

        # Diffusion moves
        step = []
        for diff_step in range(num_steps):
            instance_key = instance_keys[diff_step % len(instance_keys)]
            if not self.instances[instance_key].is_locked():
                instance: ProgInstance = self.instances[instance_key]
                random_node_id = instance.pick_node_id()  # Currently picks a node without a system
                if random_node_id is not None:
                    # print(instance_key)
                    # print(random_node_id)
                    # vals, sampler = instance.get_node_dict(random_node_id)
                    new_value = instance.resample_node(random_node_id)
                    # print(new_value)
                    step.append((instance_key, random_node_id))
        self.last_step = step

        # Birth Moves
        desired_births = self.init_n - len(self.instances) if self.init_n > len(self.instances) else 1.
        num_births = np.floor((dim_factor * np.random.random()) * (desired_births * temperature))
        self.last_birthed_instances = []
        for birth_step in range(int(num_births)):
            birthed_instance = self.sample_instance()
            self.last_birthed_instances.append(birthed_instance.get_instance_id())

    def revert_last_step(self):
        # Revert Birth steps
        for birth_id in self.last_birthed_instances:
            self.remove_instance(birth_id)

        # Revert diffusion steps
        for f_id, n_id in self.last_step:
            self.instances[f_id].revert_resampling(n_id)

        # Revert death steps
        for death_instance in self.last_killed_instances:
            self.append_instance(death_instance)  # append_instance takes care of setting the new instance id

    def _get_sampled_value(self, s: Sampler, node_id: int):
        instance = self.instances[self.last_instance_invoked]  # When executing an instance, make sure to first set this

        assert node_id is not None
        res = instance.get_value(node_id)
        if res is None:
            res = instance.set_value(node_id, s.resample(), s)

        return res

    def set_samplers(self, samplers: List[Sampler]):
        for sampler in samplers:
            sampler.delegate = functools.partial(self._get_sampled_value, sampler)
        self.samplers = samplers

    def _parse_progs(self):
        if self.words is None:
            raise Exception("ProgPool is not correctly initialized")

        roots = []
        self.dsl_fs = []
        for word, depth in self.words:
            j_word: Dict = json.loads(word)
            assert len(j_word.keys()) == 1
            # n_sem, n_type = self.semantics[list(j_word.keys())[0]]
            c_node = CNode.from_json(j_word, self._dsl.semantics)
            roots.append(c_node)

            dsl_f = DslFunc(c_node)
            if dsl_f.n_nodes <= self.max_nodes:
                self.dsl_fs.append(dsl_f)

        # self.dsl_fs = [DslFunc(n) for i, n in enumerate(roots)]
        # print(self.dsl_fs)

    def sample_words(self, max_n=10) -> None:
        words = self._dsl.sample_n_words(self.r_type, max_n=max_n)
        self.words = words
        self._parse_progs()

    def __getitem__(self, item: int):
        return self.dsl_fs[item]

    def append_instance(self, new_instance: ProgInstance):
        new_instance.set_instance_id(self.new_instance_enumerator)
        self.instances[self.new_instance_enumerator] = new_instance
        self.new_instance_enumerator += 1

    def initialize_instances(self):
        if self.init_n > len(self.dsl_fs):
            print(f"There were less programs {len(self.dsl_fs)} than instances requested {self.init_n}")
        for i in range(min(self.init_n, len(self.dsl_fs))):
            new_instance = ProgInstance(i)
            self.append_instance(new_instance)

    def visualize_instance(self, instance_id: int, parent_dir='', f_name='') -> None:
        if instance_id in self.instances:
            instance = self.instances[instance_id]
            dsl_func = self.dsl_fs[instance.get_architecture_id()]
            dsl_func.visualize(f_name, dir_path=parent_dir, instance_id=instance_id)
        else:
            lib.logger.debug_var(f"Instance {instance_id} cannot be visualized, because it doesnt exist")

    def sample_instance(self) -> ProgInstance:
        archi_id = random.randint(0, len(self.dsl_fs) - 1)
        new_instance = ProgInstance(archi_id)
        self.append_instance(new_instance)
        return new_instance

    def duplicate_instance(self, instance_id: int):
        new_instance = self.instances[instance_id].duplicate()
        self.append_instance(new_instance)

    def remove_instance(self, instance_id: int, p_locked=0.9) -> Optional[ProgInstance]:
        """
        Removes an instance from the pool (Death Move).
        """
        if instance_id in self.instances:
            if not self.instances[instance_id].is_locked():
                inst = self.instances.pop(instance_id)
                return inst
            else:
                # Instance is locked:
                if random.random() < p_locked:
                    self.instances[instance_id].unlock()
                    return None

    def remove_unused(self, remove_all_zero=True):
        usage = np.array([self.instances[key].stats['used'] for key in self.instances])
        candidates = np.where(usage == np.min(usage))[0]
        removal_id = np.random.choice(candidates)
        self.remove_instance(list(self.instances.keys())[removal_id])

    def execute(self, instance_id: int, state: Optional[Dict] = None, store=True, visualize_after=False):
        self.last_instance_invoked = instance_id
        instance: ProgInstance = self.instances[instance_id]
        dsl_func = self.dsl_fs[instance.get_architecture_id()]
        if state is not None:
            self._dsl.context.set_state(state)
        res = dsl_func.execute(store_result=store)

        if visualize_after:
            dsl_func.visualize(instance_id=instance_id)

        return res

    def compute_features(self, states: List[Dict], store=True, visualize_after=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Given a list of states, computes the output for all given states.
        Output: (#ProgramInstances, #states, Width, Height)
        """
        res = []
        i_ids = []
        for instance_id in self.instances:
            state_results = []
            for state in states:
                state_res = self.execute(instance_id, state, store=store, visualize_after=visualize_after)
                state_results.append(state_res)
            res.append(np.array(state_results))
            i_ids.append(instance_id)
        r = np.array(res), np.array(i_ids)
        return r
