"""
This file stores scoring functions for model evaluation.
"""
# -------------------------------------------------------------------------------------------------------------------- #

# Standard Library

# Third Party
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

# Private


# -------------------------------------------------------------------------------------------------------------------- #


def mse(y, pred):
    return mean_squared_error(y, pred)


def mae(y, pred):
    return mean_absolute_error(y, pred)


def r2(y, pred):
    return r2_score(y, pred)


def corr(y, pred):
    return np.corrcoef(y, pred)[0, 1]


def acc(y, pred, threshold):
    acc = sum((abs(pred - y) < threshold)) / len(pred)
    return acc
