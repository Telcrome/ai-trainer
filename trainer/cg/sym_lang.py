from __future__ import annotations
import itertools
import random
from enum import Enum
from abc import ABC
from typing import TypeVar, NewType, Union, Callable, Tuple, Any, Dict, get_type_hints, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import trainer.lib as lib
from trainer.demo_data.arc import game_from_subject, Value
from trainer.cg.Dsl import Dsl, Context, ProgPool
from trainer.cg.dsl_utils import colour_converter
from trainer.cg.samplers import FloatSampler, EnumSampler, RandomNumber, NumberSampler
from trainer.cg.typed_dsl import *


class ArcContext(Context):
    def sit(self) -> ValueGrid:
        return self.state['grid']

    def get_colours(self) -> List[Value]:
        return [colour_converter[v] for v in self.state['values']]


def arc_specifics():
    c = ArcContext()
    common = [
        # Value
        (pick_from_values, 2.),

        # List[Value]
        (c.get_colours, 5.),
        (sorted_values, 1.),

        # ValueGrid
        (c.sit, 5.),
        (shift_val_arr, 1.),
        (tiled, 0.8),
        (transform, 1.),
        (zoom_valgrid, 0.8),
        (obj_to_valgrid, 0.9),

        # BoolGrid
        (is_value, 1.1),
        (negated_arr, 0.4),
        (zoom_boolgrid, 0.5),
        (obj_to_boolgrid, 0.8),

        # RealGrid
        (reg_quantity, 1.),
        (bool_to_real, 2.),
        (int_to_real, 1.),

        # IntGrid
        (apply_boolf, 1.),
        (ident_neigh, 0.9),
        (arr_from_line, 1.),
        (hist, 1.),

        # IntLine
        (coord, 1.),
        (different_cells_in_line, 1.1),

        # ObjectLabels
        (lbl_connected, 1.),
        (lbl_by_bg, 1.),
        (lbl_by_bool, 0.8),

        # SingleObject
        (move_to, 1.),
        (object_by_ordering, 1.),
        (object_by_spatial, 1.),

        # BoolFilter
        (filter_3x3, 1.),

        # PositiveNumber
        (measure_grid, 1.),
        (measure_grid_b, 0.9),

        # NonZeroNumber
        (non_zero_num, 1.),

        # Offset
        (direction_step, 1.),
        (make_offset, 0.5),

        # Position
        (origin, 2.),
        (get_obj_lu, 1.)
    ]
    feature_sems = common + []
    action_sems = common + [
        # ValueGrid
        (value_to_arr, 1.),
    ]
    enums = [B, Value, OneShotTransform, RegionProp, Structs, RFilters, Orientation]
    feature_samplers = [EnumSampler(e) for e in enums] + [NumberSampler(1, 10, PositiveNumber)]
    action_samplers = [EnumSampler(e) for e in enums] + [NumberSampler(1, 10, PositiveNumber)]
    return c, feature_sems, feature_samplers, action_sems, action_samplers


def load_pps_from_disk(f_csv_path: str, a_csv_path: str) -> Tuple[ProgPool, ProgPool]:
    c, f_sems, f_samplers, a_sems, a_samplers = arc_specifics()
    pp_features = ProgPool.from_disk(f_csv_path, RealGrid, f_sems, c, f_samplers)
    pp_actions = ProgPool.from_disk(a_csv_path, ValueGrid, a_sems, c, a_samplers)
    return pp_features, pp_actions


def get_pps(max_f=200, max_a=100) -> Tuple[ProgPool, ProgPool]:
    c, f_sems, f_samplers, a_sems, a_samplers = arc_specifics()

    pp_features: ProgPool = ProgPool(RealGrid, f_sems, c, f_samplers)
    pp_actions: ProgPool = ProgPool(ValueGrid, a_sems, c, a_samplers)
    pp_features.sample_words(max_f)
    pp_actions.sample_words(max_a)
    return pp_features, pp_actions


if __name__ == '__main__':
    sess = lib.Session()
    test_subject = sess.query(lib.Subject).filter(lib.Subject.name == '007bbfb7').first()
    # test_subject = sess.query(lib.Subject).filter(lib.Subject.name == 'd23f8c26').first()
    game = game_from_subject(test_subject)
    train_states, test_states = game.get_states()

    print('Starting')
    _, pp = get_pps()
    pp.set_init_num(100)

    pp.initialize_instances()
    print("Ready to go")

    # pp.duplicate_instance(0)
    res = pp.compute_features([train_states[0]], visualize_after=True)
    print("Computed features")
    try:
        res = pp.compute_features([train_states[0]], visualize_after=True)
        print("Computed features")
    except Exception as e:
        print(e)

    # pp.mcmc_step()
    #
    # res2 = pp.compute_features(train_states, visualize_after=True)
    #
    # pp.revert_last_step()
    #
    # res3 = pp.compute_features(train_states, visualize_after=True)

    # for key in pp.instances:
    #     pp.execute(pp.instances[key].get_instance_id(), test_state, visualize_after=True)

    # for pt in pp:
    #     pp.visualize_execution(pt, test_state)

    # pp.mcmc_step()
    #
    # for pt in pp:
    #     pp.visualize_execution(pt, {'val': Value.Red})
    #
    # pp.revert_last_step()
    #
    # for pt in pp:
    #     pp.visualize_execution(pt, {'val': Value.Red})

    # pp.visualize_execution(pp[0], {'val': Value.Red})

    # x = dsl.execute(pp[0], {'val': Value.Red})

    # for pt in pp:
    #     # pt.visualize()
    #     dsl.visualize_execution(pt, {'val': Value.Empty})
    #
    # print(dsl.grammar.prod_rules)
