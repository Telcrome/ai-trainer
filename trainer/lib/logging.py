from __future__ import annotations

import datetime
from typing import Any, List, Optional
import logging
import os
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sqlalchemy as sa
from sqlalchemy.orm import relationship
from trainer.db import Base

import trainer.lib as lib

TABLENAME_EXPERIMENT_RESULT = 'experimentresults'
TABLENAME_EXPERIMENT = 'experiments'


class LogWriter:

    def __init__(self, log_dir: str = './logs', id_hint='log'):
        self.prepped = False
        self.log_dir, self.log_id, self.logger = log_dir, lib.create_identifier(hint=id_hint), None

    def _prep(self):
        if not self.prepped:
            if not os.path.exists(self.log_dir):
                os.mkdir(self.log_dir)
            if not os.path.exists(self._get_run_path()):
                os.mkdir(self._get_run_path())

            # Logging to file
            self.logger = logging.getLogger('spam')
            self.logger.setLevel(logging.DEBUG)
            fh = logging.FileHandler(os.path.join(self._get_run_path(), 'logs.txt'))
            fh.setLevel(logging.DEBUG)
            self.logger.addHandler(fh)
            self.logger.propagate = False
            logging.info(self.log_id)
        self.prepped = True

    def _log_str(self, c: str):
        self._prep()
        self.logger.info(c)

    def _save_fig(self, fig: plt.Figure, title='fig', close=True, max_file_path_len=140):
        self._prep()

        f_name = f"{lib.slugify(title)}"[:max_file_path_len]
        if f_name in os.listdir(self._get_run_path()):
            f_name = f'{f_name}_{len(os.listdir(self._get_run_path()))}'
        fig.savefig(os.path.join(self._get_run_path(), f'{f_name}.png'))
        if close:
            plt.close(fig)

    def _debug_arr(self, arr: np.ndarray) -> plt.Figure:
        self._log_str(f'{arr.shape}, {arr.dtype}. Values: {np.unique(arr, return_counts=True)}')
        if len(arr.shape) == 2:
            fig, ax = plt.subplots()
            sns.heatmap(arr.astype(np.float32), ax=ax)
            return fig

    def debug_var(self, o: Any) -> None:
        """
        Allows to inspect an arbitrary python object on disk.

        For saving an array with a description debug a (np.ndarray, str) tuple.

        :param o: Any variable
        """
        if isinstance(o, tuple):
            if isinstance(o[0], np.ndarray) and isinstance(o[1], str):
                fig = self._debug_arr(o[0])
                fig.suptitle(o[1])
                self._save_fig(fig, title=o[1])
            else:
                for item in o:
                    self.debug_var(item)
        elif isinstance(o, np.ndarray):
            fig = self._debug_arr(o)
            self._save_fig(fig)
        elif isinstance(o, str):
            self._log_str(f'\n{o}\n')
        else:
            self._log_str(str(o))

    def _get_run_path(self) -> str:
        return os.path.join(self.get_parent_log_folder(), self.log_id)

    def get_parent_log_folder(self) -> str:
        # self._prep()
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        return self.log_dir

    def get_absolute_run_folder(self) -> str:
        # self._prep()
        if not os.path.exists(self._get_run_path()):
            os.mkdir(self._get_run_path())
        return self._get_run_path()


logger = LogWriter()


class ExperimentResult(Base):
    """
    The semantics of one instance of this class might be: data point #223 was correctly classified.
    """
    __tablename__ = TABLENAME_EXPERIMENT_RESULT

    id = sa.Column(sa.Integer, primary_key=True)
    exp_id = sa.Column(sa.Integer, sa.ForeignKey(f'{TABLENAME_EXPERIMENT}.id'))
    name = sa.Column(sa.String())
    flag = sa.Column(sa.String())

    @classmethod
    def build_new(cls, name: str, flag: str):
        res = cls()
        res.name, res.flag = name, flag
        return res


class Experiment(Base):
    """
    Allows to track solutions over time. Uses the database for different types of logs.
    """
    __tablename__ = TABLENAME_EXPERIMENT

    id = sa.Column(sa.Integer, primary_key=True)
    experiment_name = sa.Column(sa.String())
    start_date = sa.Column(sa.DateTime())

    results: List[ExperimentResult] = relationship(ExperimentResult)

    @classmethod
    def build_new(cls, experiment_name: str):
        res = cls()
        res.experiment_name = experiment_name
        res.start_date = datetime.datetime.utcnow()
        return res

    def add_result(self, result_name: str, flag='success') -> None:
        """
        Add a new result to the current run.

        :param result_name: Value of the result.
        :param flag: Flag of the result. For example 'success' or 'fail'. It is case-sensitive.
        """
        exp_res = ExperimentResult.build_new(result_name, flag)
        self.results.append(exp_res)

    def is_in(self, result_name: str, flag='success') -> bool:
        raise NotImplementedError()

    def get_results(self, flag='success') -> List[str]:
        return [exp_res.name for exp_res in self.results if exp_res.flag == flag]

    @staticmethod
    def get_all_with_flag(sess: sa.orm.session.Session, exp_name: str, flag='success') -> List[str]:
        """
        Computes all results with a certain flag from the history i.g. all runs.
        """
        exps = sess.query(Experiment).filter(Experiment.experiment_name == exp_name)

        # res = []
        # for run_key in self.json_content:
        #     if flag in self.json_content[run_key]:
        #         for res_name in self.json_content[run_key][flag]:
        #             if res_name not in res:
        #                 res.append(res_name)
        # return res

    def __repr__(self) -> str:
        res = f'Progress tracker with {len(self.results)} runs\n'
        for exp_res in self.results:
            res += f'Current run: {exp_res}'
        return res
