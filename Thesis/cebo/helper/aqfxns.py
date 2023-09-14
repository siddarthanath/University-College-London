"""
This file stores acquisition functions for Bayesian Optimisation.
"""
# -------------------------------------------------------------------------------------------------------------------- #

# Standard Library
from typing import Union

# Third Party
import numpy as np
from scipy.stats import norm

# Private
from ..helper.distmodel import DiscreteDist, GaussDist


# -------------------------------------------------------------------------------------------------------------------- #


def expected_improvement(dist: Union[DiscreteDist, GaussDist], best: float):
    if isinstance(dist, DiscreteDist):
        ei = np.sum(np.maximum(dist.values - best, 0) * dist.probs)
        return ei
    elif isinstance(dist, GaussDist):
        z = (dist.mean() - best) / dist.std()
        ei = (dist.mean() - best) * norm.cdf(z) + dist.std() * norm.pdf(z)
        return ei


def probability_of_improvement(dist, best):
    if isinstance(dist, DiscreteDist):
        pi = np.sum(np.cast[float](dist.values > best) * dist.probs)
        return pi
    elif isinstance(dist, GaussDist):
        z = (dist.mean() - best) / dist.std()
        pi = norm.cdf(z)
        return pi


def upper_confidence_bound(dist, _lambda):
    if isinstance(dist, DiscreteDist):
        mu = np.sum(dist.values * dist.probs)
        sigma = np.sqrt(np.sum((dist.values - mu) ** 2 * dist.probs))
        return mu + _lambda * sigma
    elif isinstance(dist, GaussDist):
        return dist.mean() + _lambda * dist.std()


def greedy(dist):
    if isinstance(dist, DiscreteDist):
        return dist.values[np.argmax(dist.probs)]
    elif isinstance(dist, GaussDist):
        return dist.mean()


def thompson_sampling(dist, probs, values):
    if isinstance(dist, DiscreteDist):
        return np.random.choice(values, p=probs)
    elif isinstance(dist, GaussDist):
        return np.random.normal(loc=dist.mean(), scale=dist.std())
