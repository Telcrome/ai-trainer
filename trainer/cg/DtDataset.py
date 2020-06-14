"""
A module which contains glue code.

Loads two program pools from disk. One for computing features, one for exploring actions.
Then the pure output of the sampled programs is computed for a given game.


Actions are scored by the number of cells that can be explained by that specific action.

"""
from typing import List, Tuple, Generator, Any, Iterable, Dict, Callable, Union, Optional
import os
import itertools

from tqdm import tqdm
import numpy as np
import pandas as pd
import graphviz
import sklearn.tree as tree
import matplotlib.pyplot as plt
import seaborn as sns

import trainer.lib as lib
import trainer.ml as ml

from trainer.demo_data.arc import Pair, Game, game_from_subject
from trainer.cg.ProgPool import ProgPool


def vis_dec_tree(dec_tree: tree.DecisionTreeClassifier, name: str, prog_names: np.ndarray, class_names: List[str],
                 dir_path=''):
    if not dir_path:
        dir_path = lib.logger.get_absolute_run_folder()
        
    dot_data = tree.export_graphviz(dec_tree, out_file=None,
                                    feature_names=prog_names,
                                    class_names=class_names,
                                    filled=True,
                                    rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    p = os.path.join(dir_path, lib.slugify(name))
    # lib.logger.debug_var(f'Rendering at {p}')
    try:
        # render('dot', 'png', f_path)
        graph.render(p, format='png')
    except Exception as e:
        print(e)


Dataset = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
DtData = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
GameData = Tuple[DtData, DtData]


class DtDataset:
    def __init__(self, pp_f: ProgPool, pp_a: ProgPool):
        self.feature_pp, self.actions_pp = pp_f, pp_a

    def _compute_raw_data(self, states: List[Dict], vis=False, store=False) -> Dataset:
        features, f_consts = self.feature_pp.compute_features(states, visualize_after=vis, store=store)
        a_values, a_consts = self.actions_pp.compute_features(states, visualize_after=vis, store=store)

        actions = np.zeros_like(a_values)
        for a_id in range(a_values.shape[0]):
            for s_i, state in enumerate(states):
                actions[a_id, s_i] = a_values[a_id, s_i] == state['target']
        return features, f_consts, actions, a_consts, a_values

    @staticmethod
    def build_dt_set(raw_data: Dataset) -> DtData:
        features, f_names, actions, a_names, a_values = raw_data
        flat_features = np.vstack([features[i].flatten() for i in range(features.shape[0])]).transpose()
        flat_actions = np.vstack([actions[i].flatten() for i in range(actions.shape[0])]).transpose()
        flat_vals = np.vstack([a_values[i].flatten() for i in range(a_values.shape[0])]).transpose()
        return flat_features, f_names, flat_actions, a_names, flat_vals

    def get_data(self, game: Game) -> GameData:
        train_states, test_states = game.get_states()
        train_set, test_set = self._compute_raw_data(train_states), self._compute_raw_data(test_states)
        train_set, test_set = DtDataset.build_dt_set(train_set), DtDataset.build_dt_set(test_set)
        return train_set, test_set
