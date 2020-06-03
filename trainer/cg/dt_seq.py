from typing import List

import numpy as np


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


class DtSeq:
    """
    Encapsulates a DtSeq model, public methods inspired from sklearn.
    """

    def __init__(self):
        pass

    def fit(self, train_x: np.ndarray, train_y: np.ndarray, train_vals: np.ndarray) -> None:
        assert train_x.shape[0] == train_y.shape[0] == train_vals.shape[0]
        assert train_y.shape[1] == train_vals.shape[1]
