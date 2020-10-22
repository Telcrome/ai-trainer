"""
Solver that computes a progpool with programs that should be used as a starting point for predictions.
Meant as entry code that glues the other components together.
"""
import os
import itertools
import random
from typing import List, Optional, Dict, Tuple

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import trainer.lib as lib
import trainer.ml as ml

from trainer.demo_data.arc import plot_as_heatmap, game_from_subject, Game

from trainer.cg.sym_lang import get_pps, load_pps_from_disk
from trainer.cg.DtDataset import DtDataset
from trainer.cg.dt_seq import ArcTransformation, pred_equal

EXP_NAME = 'cg_with_generalization_cheat'
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
# STEP_GENERALIZE_PROBABILITY = 0.2
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


def load_subject_dataset(g: Game):
    train_set, test_set = dt_set.get_data(g)
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
    tracker = lib.Experiment.build_new(EXP_NAME, sess=sess)
    prev_solved = tracker.get_all_with_flag(sess, EXP_NAME, flag='success')

    arc_ds: lib.Dataset = sess.query(lib.Dataset).filter(lib.Dataset.name == 'arc').first()

    train_split: lib.Split = arc_ds.get_split_by_name('training')
    eval_split: lib.Split = arc_ds.get_split_by_name('evaluation')
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
    # unsolved = train_split.sbjts
    unsolved = train_split.sbjts
    random.shuffle(unsolved)

    solutions: Dict[str, List[np.ndarray]] = {}

    for epoch in [0]:  # itertools.count():
        epoch_sol_found, epoch_successes, epoch_fails = [], [], []
        epoch_logdir = prepare_epoch_logging()
        f_path = feature_pp.to_disk(f"features_epoch{epoch}", parent_dir=epoch_logdir)
        a_path = actions_pp.to_disk(f"actions_epoch{epoch}", parent_dir=epoch_logdir)

        unsolved = [s for s in unsolved if s.name not in tracker.get_results(flag='success')]
        pbar = tqdm(unsolved)
        for epoch_step, s in enumerate(pbar):
            nodes_current, temperature, all_preds = NO_SOLUTION_NODE_COUNT, 1.0, []
            feature_pp.instances = {}
            actions_pp.instances = {}
            feature_pp.initialize_instances()
            actions_pp.initialize_instances()

            # A list for each time solutions were found
            prediction_history: List[Tuple] = []
            g: Game = game_from_subject(s)

            for mcmc_step in range(MAX_MCMC_STEPS):
                # TODO do not reload game information at every step?
                x, y, vals, x_test, y_test, vals_test, fs, acts = load_subject_dataset(g)
                data_sanity_check()  # Simple shape check for the dataset

                # Update the progress bar with most relevant information
                update_s = f"Step: {mcmc_step}; Latest: {tracker.get_results(flag='success')[-3:]};"
                update_s += f"temperature: {temperature}; "
                update_s += f"Consistent: {len(set(epoch_sol_found))}/{epoch_step}, N: {nodes_current}; "
                update_s += f'F-Locks: {feature_pp.get_locks()} {x.shape}, A-Locks: {actions_pp.get_locks()} {y.shape}'
                update_s += f'Solved {len(set(tracker.get_results(flag="success")))}, {len(set(epoch_successes))}; '
                update_s += f'new: {str([x for x in tracker.get_results(flag="success") if x not in prev_solved])}'
                pbar.set_postfix_str(update_s)

                sols: Optional[List[ArcTransformation]] = ArcTransformation.find_consistent_solutions(x, y, fs, acts,
                                                                                                      max_steps=10)

                # Record statistics to find leeches that have no or little predictive value
                for sol in sols:
                    steps = sol.get_used_cgs()
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
                            # unique_predictions.append(preds)
                            unique_sols.append(sols[i])
                            prediction_history.append((preds, sols[i]))

                    node_counts = np.array([sol.get_node_count() for sol in unique_sols])

                    # Simple heuristic for picking the best three approaches: Pick those using least number of nodes
                    check_out_order = np.argsort(node_counts)[:3]
                    # Store for submission
                    # unique_predictions.append((all_preds, node_counts))

                    unique_gen_result = [sol.test_generalization(x_test, y_test) for sol in unique_sols]
                    generalizing = np.array([t[0] for t in unique_gen_result])

                    # Train the program pools
                    all_gen_res = [sol.test_generalization(x_test, y_test) for sol in sols]

                    # Even without a sane and unique prediction, all_gen_res can be used to update the program pool
                    # As long as a consistent solution exists
                    # all_gen = [t[0] for t in all_gen_res]
                    # all_fb = [t[1] for t in all_gen_res]  # all_feedback

                    if True in generalizing:  # For testing purposes
                        p = f'{s.name}: {sum(generalizing)} of {len(unique_sols)} generalized'
                        lib.logger.debug_var(p)

                        # If this is a top 3 success solution:
                        if True in generalizing[check_out_order]:
                            epoch_successes.append(s.name)

                            # If this was the first time it was solved, visualize it
                            if not tracker.is_in(s.name, flag='success'):
                                # Visualize solutions
                                for i, sol in enumerate(sols[:MAX_VIS]):
                                    sol.visualize(parent_path=epoch_logdir, f_vis=feature_pp.visualize_instance,
                                                  a_vis=actions_pp.visualize_instance,
                                                  folder_appendix=str(all_gen_res[i][0]),
                                                  name=s.name)

                            tracker.add_result(s.name, flag='success', sess=sess)
                            break
                    else:
                        lib.logger.debug_var(f'No generalizing solution was found for {s.name}')
                        lib.logger.debug_var(f'{s.name} has {len(unique_sols)} solutions which do not generalize')
                        tracker.add_result(s.name, flag='nongeneralizing', sess=sess)
                else:
                    node_counts, check_out_order = np.array([]), np.array([])

                # Diffusion Move
                # If the sanity check rules out all predictions, node_counts will be empty
                if node_counts.shape[0] > 0:
                    nodes_proposal = np.mean(node_counts)
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

                feature_pp.optim_move(temperature=temperature)
                feature_pp.visualize_instance(0)
                actions_pp.optim_move(temperature=temperature)

            node_counts_candidates = [sol.get_node_count() for _, sol in prediction_history]
            pred_history = [preds for preds, _ in prediction_history]
            indices = np.argsort(node_counts_candidates)

            # Use 'least node count' heuristic to find the solution to submit for each test pair
            submission = {test_i: [] for test_i in range(len(g.test_pairs))}
            for test_i, test_pair in enumerate(g.test_pairs):
                for sol_index in indices:  # Add prediction in the order given by the heuristic
                    if len(submission[test_i]) >= 3:
                        break
                    arr = pred_history[sol_index][test_i]
                    equal_old_preds = [np.array_equal(arr, candidate) for candidate, _ in submission[test_i]]
                    if not (True in equal_old_preds):
                        submission[test_i].append((arr, prediction_history[sol_index][1]))

            dir_name = os.path.join(epoch_logdir, f'{s.name}')
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            for test_key in submission:
                one_generalized = False
                for pred_i, (arr, sol) in enumerate(submission[test_key]):
                    generalized = np.array_equal(arr, g.test_pairs[test_key].get_target())
                    one_generalized = one_generalized or generalized
                    fig, ax = plt.subplots()
                    plot_as_heatmap(arr, ax=ax, title=f'{s.name}: {sol.get_node_count()}')
                    fig.savefig(os.path.join(dir_name, f'{test_key}_{pred_i}_{generalized}.png'))
                    plt.close(fig)
                if one_generalized:
                    if s.name not in epoch_successes:
                        tracker.add_result(s.name, flag='success', sess=sess)
                        epoch_successes.append(s.name)
                else:
                    if s.name not in epoch_fails:
                        tracker.add_result(s.name, flag='fail', sess=sess)
                        epoch_fails.append(s.name)

            # candidates = []
            # for test_i in range(len(node_counts_candidates)):
            #     indices = np.argsort(node_counts_candidates[test_i])
            #     candidates_i: List[np.ndarray] = []
            #     for pred, node_n in unique_predictions[test_i]:
            #         if len(candidates_i) >= 3:
            #             break
            #         equal_old_preds = [np.array_equal(pred, candidate) for candidate, _ in candidates_i]
            #         if not (True in equal_old_preds):
            #             candidates_i.append((pred, node_n))
            #     candidates.append(candidates_i)
            # solutions[s.name] = candidates
            # feature_pp, actions_pp = load_pps_from_disk(f'{f_path}.xlsx', f'{a_path}.xlsx')
