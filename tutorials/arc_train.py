"""
Solver that computes a progpool with programs that should be used as a starting point for predictions.
Meant as entry code that glues the other components together.
"""
import os
import itertools
import random
from typing import List, Optional

from tqdm import tqdm
import numpy as np

import trainer.lib as lib
import trainer.ml as ml

from trainer.demo_data.arc import plot_as_heatmap, game_from_subject

from trainer.cg.sym_lang import get_pps, load_pps_from_disk
from trainer.cg.Dsl import ProgPool
from trainer.cg.DtDataset import DtDataset
from trainer.cg.dt_seq import ArcTransformation, pred_equal

LOAD_FROM_DISK = False
PLOT_PREDICTIONS = False
MAX_PLOTS = 5
MAX_VIS = 5
FEATURE_N = 400
ACTION_N = 200
START_F_N = 100
START_A_N = 50
MAX_MCMC_STEPS = 100

# If only a substep generalizes but not the whole solution, this is the probability for a birth move
COOLING_RATE = 0.98
STEP_GENERALIZE_PROBABILITY = 0.2
DEATH_ON_FAIL_PROBABILITY = 0.9
BIRTH_PRIOR = 0.5
DEATH_PRIOR = 0.5

NO_SOLUTION_NODE_COUNT = 10000


def arc_sanity_check(prediction: np.ndarray) -> bool:
    """
    Tests simple criteria for an array that need to be fulfilled to be an ARC-Grid
    """
    non_empty = np.max(prediction) > 0
    if not non_empty:
        return False
    reduced, reducedatt, (l, r, b, t) = ml.reduce_by_attention(prediction, prediction != 0)
    correct_grid_shape = 0 not in reduced

    return non_empty and correct_grid_shape


def load_subject_dataset():
    game = game_from_subject(s)
    train_set, test_set = dt_set.get_data(game)
    x_train, f_insts, y_train, a_insts, vals_train = train_set
    x_test, _, y_test, _, vals_test = test_set
    return x_train, y_train, vals_train, x_test, y_test, vals_test, f_insts, a_insts


def prepare_epoch_logging():
    p = os.path.join(lib.logger.get_absolute_run_folder(), f'epoch_{epoch}')
    if not os.path.exists(p):
        os.mkdir(p)
    return p


def shrink_pool(pp, n, thres=1.2):
    if pp.instances.__len__() > n * thres:
        for _ in range(random.randint(1, pp.instances.__len__() - n)):
            pp.remove_unused()


def data_sanity_check():
    err_msg = "Number of features might be dependent on situation"
    assert x.shape[1] == x_test.shape[1] and y.shape[1] == y_test.shape[1], err_msg
    assert x.shape[0] == y.shape[0] and x_test.shape[0] == y_test.shape[0]


if __name__ == '__main__':
    # regenerate_programs(test_programs=False)
    sess = lib.Session()
    # TODO write a simple module which dumps the previously solved tasks into a json and loads them here
    prev_solved = []

    train_split: lib.Split = sess.query(lib.Split).filter(lib.Split.name == 'training').first()
    eval_split: lib.Split = sess.query(lib.Split).filter(lib.Split.name == 'evaluation').first()
    interesting_now = ['de493100']

    if LOAD_FROM_DISK:
        p_features, _ = lib.standalone_foldergrab(folder_not_file=False, title='Pick Feature csv')
        p_actions, _ = lib.standalone_foldergrab(folder_not_file=False, title='Pick Action csv')
        feature_pp, actions_pp = load_pps_from_disk(p_features, p_actions)
    else:
        feature_pp, actions_pp = get_pps(max_f=FEATURE_N, max_a=ACTION_N)
    feature_pp.set_init_num(START_F_N)
    actions_pp.set_init_num(START_A_N)
    dt_set = DtDataset(feature_pp, actions_pp)

    # unsolved: List[lib.Subject] = sess.query(lib.Subject).filter(lib.Subject.name.in_(prev_solved)).all()
    # unsolved: List[lib.Subject] = sess.query(lib.Subject).filter(lib.Subject.name.in_(interesting_now)).all()
    unsolved = train_split.sbjts + eval_split.sbjts
    random.shuffle(unsolved)
    successful = []
    solved_before = []

    for epoch in itertools.count():
        epoch_sol_found, epoch_successes = [], []
        epoch_logdir = prepare_epoch_logging()
        f_path = feature_pp.to_disk(f"features_epoch{epoch}", parent_dir=epoch_logdir)
        a_path = actions_pp.to_disk(f"actions_epoch{epoch}", parent_dir=epoch_logdir)

        unsolved = [s for s in unsolved if s.name not in successful]
        pbar = tqdm(unsolved)
        for epoch_step, s in enumerate(pbar):
            nodes_current, temperature = NO_SOLUTION_NODE_COUNT, 1.0
            feature_pp.instances = {}
            actions_pp.instances = {}
            feature_pp.initialize_instances()
            actions_pp.initialize_instances()

            for mcmc_step in range(MAX_MCMC_STEPS):
                # TODO do not reload game information at every step
                x, y, vals, x_test, y_test, vals_test, fs, acts = load_subject_dataset()
                data_sanity_check()  # Simple shape check for the dataset

                # Update the progress bar with most relevant information
                update_s = f"Step: {mcmc_step}; Latest: {successful[-3:]}; temperature: {temperature}; "
                update_s += f"Consistent: {len(set(epoch_sol_found))}/{epoch_step}, N: {nodes_current}; "
                update_s += f'F-Locks: {feature_pp.get_locks()} {x.shape}, A-Locks: {actions_pp.get_locks()} {y.shape}'
                update_s += f'Solved {len(set(successful))}, {len(set(epoch_successes))}; '
                update_s += f'new: {str([x for x in successful if x not in prev_solved])}'
                pbar.set_postfix_str(update_s)

                sols: Optional[List[ArcTransformation]] = ArcTransformation.find_consistent_solutions(x, y, fs, acts,
                                                                                                      max_steps=10)

                # Record statistics to find leeches that have no or little predictive value
                for sol in sols:
                    steps = sol.get_used_fs_a()
                    for step, (a_instance_id, f_instance_ids) in enumerate(steps):
                        actions_pp.instances[a_instance_id].increment_used()
                        for f_instance_id in f_instance_ids:
                            feature_pp.instances[f_instance_id].increment_used()

                if sols:
                    epoch_sol_found.append(epoch_step)

                    all_preds: List[List[np.ndarray]] = [sol.predict(x_test, vals_test) for sol in sols]
                    # sols[0].visualize(parent_path=epoch_logdir, f_vis=feature_pp.visualize_instance,
                    #                   a_vis=actions_pp.visualize_instance)

                    if PLOT_PREDICTIONS:
                        for preds in all_preds[:MAX_PLOTS]:
                            for pred in preds:
                                plot_as_heatmap(pred, title=s.name)

                    # Filter the predictions by uniqueness (to not submit identical predictions) and a sanity check
                    unique_predictions, unique_sols = [], []
                    for i, preds in enumerate(all_preds):
                        already_exists = [pred_equal(preds, preds_old) for preds_old in unique_predictions]
                        sanity_checks = [arc_sanity_check(pred) for pred in preds]
                        if not (True in already_exists) and not (False in sanity_checks):
                            unique_predictions.append(preds)
                            unique_sols.append(sols[i])

                    node_counts = np.array([sol.get_node_count() for sol in unique_sols])

                    # Simple heuristic for picking the best three approaches: Pick those using least number of nodes
                    check_out_order = np.argsort(node_counts)[:3]

                    unique_gen_result = [sol.test_generalization(x_test, y_test) for sol in unique_sols]
                    generalizing = np.array([t[0] for t in unique_gen_result])

                    # Train the program pools
                    all_gen_res = [sol.test_generalization(x_test, y_test) for sol in sols]

                    all_gen = [t[0] for t in all_gen_res]
                    all_fb = [t[1] for t in all_gen_res]

                    if True in generalizing:  # For testing purposes
                        p = f'{s.name}: {sum(generalizing)} of {len(unique_sols)} generalized'
                        lib.logger.debug_var(p)
                        if True in generalizing[check_out_order]:
                            if s.name not in successful:
                                successful.append(s.name)
                                epoch_successes.append(s.name)

                                # Visualize solutions
                                for i, sol in enumerate(sols[:MAX_VIS]):
                                    sol.visualize(parent_path=epoch_logdir, f_vis=feature_pp.visualize_instance,
                                                  a_vis=actions_pp.visualize_instance,
                                                  folder_appendix=str(all_gen_res[i][0]),
                                                  name=s.name)

                                break  # TODO check if this makes sense
                    else:
                        lib.logger.debug_var(f'No generalizing solution was found for {s.name}')
                        lib.logger.debug_var(f'{s.name} has {len(unique_sols)} solutions which do not generalize')
                        # return False
                else:
                    node_counts, check_out_order = np.array([]), np.array([])

                # Diffusion Move
                # If the sanity check rules out all predictions, node_counts will be empty
                if node_counts.shape[0] > 0:
                    nodes_proposal = np.mean(node_counts[check_out_order])
                    temperature *= COOLING_RATE
                else:
                    nodes_proposal = NO_SOLUTION_NODE_COUNT
                acceptance_proba = nodes_current / nodes_proposal
                if np.random.random() < acceptance_proba:
                    # Accept
                    nodes_current = nodes_proposal
                else:
                    # Revert changes made by diffusion move
                    feature_pp.revert_last_step()
                    actions_pp.revert_last_step()

                feature_pp.diffusion_move(temperature=temperature)
                actions_pp.diffusion_move(temperature=temperature)

            # feature_pp, actions_pp = load_pps_from_disk(f'{f_path}.xlsx', f'{a_path}.xlsx')
