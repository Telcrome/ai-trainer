from __future__ import annotations
import os
from typing import List, Generator, Tuple, Optional, Callable, Set

import numpy as np
import sklearn.tree as tree
import matplotlib.pyplot as plt

import trainer.lib as lib
from trainer.demo_data.arc import plot_as_heatmap
from trainer.cg.DtDataset import vis_dec_tree


def get_leave_indices(dt: tree.DecisionTreeClassifier) -> np.ndarray:
    n_nodes = dt.tree_.node_count
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    children_left = dt.tree_.children_left
    children_right = dt.tree_.children_right
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        # node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if children_left[node_id] != children_right[node_id]:
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True
    return is_leaves


def get_leave_impurity(dt: tree.DecisionTreeClassifier) -> float:
    leave_impurities = dt.tree_.impurity[get_leave_indices(dt)]
    res = np.sum(leave_impurities)
    return res.item()


def pred_equal(preds1: List[np.ndarray], preds2: List[np.ndarray]) -> bool:
    """
    Simple utility for comparing two lists of numpy arrays using exact equality

    >>> pred_equal([np.ones(3)], [np.zeros(3)])
    False

    >>> pred_equal([np.zeros(3), np.ones(4)], [np.zeros(3), np.ones(4)])
    True

    """
    assert len(preds1) == len(preds2), "Different number of arrays is not allowed"
    assert not (False in [p1.dtype == p2.dtype for p1, p2 in zip(preds1, preds2)]), "Type mismatch"
    equal_res = [np.array_equal(p1, p2) for p1, p2 in zip(preds1, preds2)]
    return not (False in equal_res)


def order_actions(x: np.ndarray,
                  ys: np.ndarray,
                  initial_ignore=None) -> Generator[Tuple[int, tree.DecisionTreeClassifier], None, None]:
    """
    An action is assumed to be easily explainable if the corresponding decision tree has a low number of nodes.


    Because all decision trees are computed anyways for ordering the actions,
    the corresponding decision tree for an action is returned along with its index.

    :param x: Features  (N x #F)
    :param ys: All actions (N x #A)
    :param initial_ignore: Optional boolean array indicating which actions should be considered (#A)
    :return:
    """
    # lib.logger.debug_var(f'picking action for {x.shape} and {ys.shape}')

    if initial_ignore is None:
        initial_ignore = np.zeros(ys.shape[1], dtype=np.bool)
    else:
        assert initial_ignore.shape[0] == ys.shape[1] and initial_ignore.dtype == np.bool
    # Filter actions that explain no pixels, as they are useless anyways
    explains_number: np.ndarray = ys.sum(axis=0)
    # Ignore actions that explain nothing, are duplicated or should be ignored initially
    ignore: np.ndarray = initial_ignore | (explains_number == 0)

    clfs = [
        tree.DecisionTreeClassifier(max_depth=None).fit(x, ys[:, action_index]) if not ignore[action_index] else None
        for action_index in range(ys.shape[1])]
    node_counts = np.array([clf.tree_.node_count if clf is not None else np.inf for clf in clfs])
    pure_actions = np.array([not ignore[action_index] and get_leave_impurity(clf) == 0.
                             for action_index, clf in enumerate(clfs)], dtype=np.bool)

    arg_sorted = np.argsort(explains_number)
    arg_sorted_node_counts = np.argsort(node_counts)

    # candidates = (node_counts == min(node_counts)) & pure_actions
    explains_number[~pure_actions] = np.iinfo(np.int).min

    for n_count in sorted(np.unique(node_counts)):
        # Yield the actions with node-count n_count in the order of the number of training examples they explain

        # print(n_count)
        indices = (node_counts == n_count)
        map_table = np.argwhere(indices)[:, 0]
        actions = np.argsort(explains_number[indices])
        for action_index in actions:
            i = map_table[action_index]
            yield i, clfs[i]
            return

    # for a_index in np.flip(arg_sorted_node_counts):
    #     if pure_actions[a_index] and not ignore[a_index]:
    #         yield a_index, clfs[a_index]
    #
    #
    # a_index, final_clf = np.argmax(explains_number).item(), clfs[np.argmax(explains_number).item()]
    # yielded = []
    # if final_clf is not None:
    #     yielded.append(a_index)
    #     yield a_index, final_clf

    # for a_index in np.flip(arg_sorted)[:branching_expressiveness]:
    #     if pure_actions[a_index] and not ignore[a_index]:
    #         if a_index not in yielded:
    #             yielded.append(a_index)
    #             yield a_index, clfs[a_index]
    #


class ArcTransformation:
    """
    Stores one transformation step as a tuple (action_index, decision_tree).
    Stores a transformation as a list of those tuples.
    """

    @staticmethod
    def enumerate_consistent_moves(x: np.ndarray,
                                   y: np.ndarray,
                                   f_inst: np.ndarray,
                                   a_inst: np.ndarray,
                                   reduce_duplicated_actions=False
                                   ) -> List[Tuple[int, tree.DecisionTreeClassifier, np.ndarray, np.ndarray]]:
        """
        Enumerates all moves that might make sense given by the heuristic of pick_action

        :param reduce_duplicated_actions:
        :param a_inst:
        :param f_inst:
        :param x: Array of shape (rows, number_of_features)
        :param y: Array of shape (rows, actions)
        :return: List of (Action_index, remaining_x_rows, remaining_y_rows)
        """
        res = []
        assert x.shape[0] > 0, 'Do not enumerate_consistent_moves if the task is already consistent'

        # if reduce_duplicated_actions:
        #     ignore_by_duplicated = np.zeros(y.shape[1], dtype=np.bool)
        #     dcs = duplicate_columns(y)
        #     for cs in dcs:
        #         ignore_by_duplicated[cs[cs != a_pp.decide_between(cs, a_inst)]] = True
        #         print(f'Ignoring {np.sum(ignore_by_duplicated)} actions by default')
        # else:
        #     ignore_by_duplicated = None
        ignore_by_duplicated = None

        for a_index, dt_tree in order_actions(x, y, initial_ignore=ignore_by_duplicated):
            keep = y[:, a_index] == 0  # Keep all rows that the action last taken did not explain
            t = a_index, dt_tree, x[keep], y[keep]
            res.append(t)
        return res

    @staticmethod
    def increment_pool(pool: List[Tuple[ArcTransformation, np.ndarray, np.ndarray]], f_s, a_s
                       ) -> List[Tuple[ArcTransformation, np.ndarray, np.ndarray]]:
        res = []
        for at, x, y in pool:
            moves = ArcTransformation.enumerate_consistent_moves(x, y, f_s, a_s)
            for a_i, dt, x, y in moves:
                t = at.append_step((a_i, dt)), x, y
                res.append(t)
        return res

    @staticmethod
    def find_consistent_solutions(x: np.ndarray, y: np.ndarray, f_s: np.ndarray,
                                  a_s: np.ndarray, max_steps=10) -> Optional[List[ArcTransformation]]:
        """
        Gets called with the full dataset for one subject. Returns a list of consistent ArcTransformations for this task.
        Use max_steps for early stopping, but keep in mind that the first steps are the expensive ones.
        """
        consistent = []
        ts = ArcTransformation.enumerate_consistent_moves(x, y, f_s, a_s)
        mem = []
        for a_i, dt, x, y in ts:
            mem.append((ArcTransformation(f_s, a_s, steps=[(a_i, dt)]), x, y))

        for _ in range(max_steps):
            mem_next_step = []
            for at, x, y in mem:
                if x.shape[0] == 0:
                    consistent.append(at)
                else:
                    mem_next_step.append((at, x, y))
            mem = mem_next_step
            mem = ArcTransformation.increment_pool(mem, f_s, a_s)

        return consistent

    def __init__(self,
                 # f_pp: ProgramPool,
                 # a_pp: ProgramPool,
                 f_inst: np.ndarray,
                 a_inst: np.ndarray,
                 steps: Optional[List] = None):
        # self.s = s
        self.f_inst, self.a_inst = f_inst, a_inst
        if steps is None:
            steps = []
        self.steps: List[Tuple[int, tree.DecisionTreeClassifier]] = steps
        self.last_preds: Optional[List[np.ndarray]] = None

    def get_node_count(self):
        return sum([step[1].tree_.node_count for step in self.steps])

    def get_used_cgs(self) -> List[Tuple[int, Set[int]]]:
        feedback = []
        for step, (a_i, dt) in enumerate(self.steps):
            used_fs = np.argwhere(dt.feature_importances_ > 0).flatten()
            step_feedback = (self.a_inst[a_i], set([self.f_inst[f_id] for f_id in used_fs]))
            feedback.append(step_feedback)
        return feedback

    def append_step(self, t: Tuple[int, tree.DecisionTreeClassifier]) -> ArcTransformation:
        new_steps = self.steps + [t]
        return ArcTransformation(self.f_inst, self.a_inst, new_steps)

    def predict(self, x_test: np.ndarray, values: np.ndarray) -> List[np.ndarray]:
        """
        Given a tabular decision tree dataset, returns the predictions made by this dt_seq as images.

        If there are multiple test examples, multiple images will be in the results list.
        """
        assert x_test.shape[0] % 900 == 0 and values.shape[0] % 900 == 0
        num_states = values.shape[0] // 900
        reshaped_values = values.transpose().reshape((values.shape[1], num_states, 30, 30))

        ls = []
        for state_i in range(num_states):
            res = np.zeros((30, 30))
            res -= 1
            for step, (a_i, dt) in enumerate(self.steps):
                pred = dt.predict(x_test)
                pred_arr = pred.reshape(num_states, 30, 30).astype(np.bool)
                res[(res == -1) & pred_arr[state_i]] = reshaped_values[a_i, state_i][(res == -1) & pred_arr[state_i]]

            ls.append(res)
        self.last_preds = ls
        return ls

    def visualize(self,
                  parent_path='',
                  f_vis: Optional[Callable] = None,
                  a_vis: Optional[Callable] = None,
                  folder_appendix='',
                  name='unknown'):
        """
        Visualizes the whole sequence including the features and actions that are used by this dt sequence.

        :param parent_path:
        :param f_vis: A callable that takes a feature id and a disk location and visualizes that feature
        :param a_vis: A callable that takes an action id and a disk location and visualizes that action
        :param folder_appendix:
        :param name:
        :return:
        """
        if not parent_path:
            parent_path = lib.logger.get_absolute_run_folder()

        old_visualizations = list(filter(lambda x: name in x, os.listdir(parent_path)))
        dir_name = os.path.join(parent_path, f'./{name}_{len(old_visualizations)}_{folder_appendix}/')
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        if self.last_preds is not None:
            for pred_i, pred in enumerate(self.last_preds):
                fig, ax = plt.subplots()
                plot_as_heatmap(pred, ax=ax, title=name)
                fig.savefig(os.path.join(dir_name, f'{pred_i}.png'))
                plt.close(fig)

        for step, (a_i, dt) in enumerate(self.steps):
            vis_dec_tree(dt, f'{step}_{self.a_inst[a_i]}', self.f_inst,
                         ['Keep in pool', 'Take Action'], dir_path=dir_name)

        if f_vis is not None and a_vis is not None:
            fb = self.get_used_cgs()
            for step, (a_instance_id, f_instance_ids) in enumerate(fb):
                a_vis(a_instance_id, parent_dir=dir_name, f_name=f'action_{a_instance_id}_step_{step}.png')
                for f_instance_id in f_instance_ids:
                    f_vis(f_instance_id, parent_dir=dir_name, f_name=f'feature_{f_instance_id}_step_{step}.png')

    def test_generalization(self, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[bool, List[Tuple[int, Set[int]]]]:
        """
        Judges the generalization performance given a test set in dt-format.

        Provides a feedback object of the following form: [(action_instance_name, {f_inst})]
        """
        feedback = []
        for step, (a_i, dt) in enumerate(self.steps):

            # Compute feedback
            used_fs = np.argwhere(dt.feature_importances_ > 0).flatten()
            step_feedback = (self.a_inst[a_i], set([self.f_inst[f_id] for f_id in used_fs]))
            feedback.append(step_feedback)
            # fb_step = (self.a_inst[a_i], )

            pred = dt.predict(x_test)
            keep_test = y_test[:, a_i] == 0
            if np.array_equal(pred, y_test[:, a_i]):
                lib.logger.debug_var(f'Solving step {step} worked using {self.a_inst[a_i]}')
                x_test, y_test = x_test[keep_test], y_test[keep_test]
                # dt_set.actions_pp.increment_counter(for_prog=a_s[a_index], row_index=0)
                # record_features(dt_tree, True, f_s)
            else:
                err = f"I encountered a not generalizing solution at step {step}"
                lib.logger.debug_var(err + f'({self.a_inst[a_i]})')
                # dt_set.actions_pp.increment_counter(for_prog=a_s[a_index], row_index=1)
                # record_features(dt_tree, False, f_s)
                return False, feedback
            if x_test.shape[0] == 0:
                return True, feedback
        lib.logger.debug_var(f'Solving did not work because not everything was solved')
        return False, feedback
