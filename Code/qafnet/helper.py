"""
This file stores helper functions which are used within the general interface.
For now, these are the acquisition functions used in the Bayesian Optimisation protocol.
"""
# -------------------------------------------------------------------------------------------------------------------- #

# Standard Library
import re
from typing import *

# Third Party
import numpy as np
from scipy.stats import norm

# Private
from .llm_model import DiscreteDist, GaussDist


# -------------------------------------------------------------------------------------------------------------------- #


def expected_improvement(dist: Union[DiscreteDist, GaussDist], best: float):
    if isinstance(dist, DiscreteDist):
        """Expected improvement for the given discrete distribution"""
        ei = np.sum(np.maximum(dist.values - best, 0) * dist.probs)
        return ei
    elif isinstance(dist, GaussDist):
        """Expected improvement for the given Gaussian distribution"""
        z = (dist.mean() - best) / dist.std()
        ei = (dist.mean() - best) * norm.cdf(z) + dist.std() * norm.pdf(z)
        return ei


def probability_of_improvement(dist, best):
    if isinstance(dist, DiscreteDist):
        """Probability of improvement for the given discrete distribution"""
        pi = np.sum(np.cast[float](dist.values > best) * dist.probs)
        return pi
    elif isinstance(dist, GaussDist):
        """Probability of improvement for the given Gaussian distribution"""
        z = (dist.mean() - best) / dist.std()
        pi = norm.cdf(z)
        return pi


def upper_confidence_bound(dist, best, _lambda):
    if isinstance(dist, DiscreteDist):
        """Upper confidence bound for the given discrete distribution"""
        mu = np.sum(dist.values * dist.probs)
        sigma = np.sqrt(np.sum((dist.values - mu) ** 2 * dist.probs))
        return mu + _lambda * sigma
    elif isinstance(dist, GaussDist):
        """Upper confidence bound for the given Gaussian distribution"""
        return dist.mean() + _lambda * dist.std()


def greedy(dist, best):
    if isinstance(dist, DiscreteDist):
        """Greedy selection (most likely point) for the given discrete distribution"""
        return dist.values[np.argmax(dist.probs)]
    elif isinstance(dist, GaussDist):
        """Greedy selection (most likely point) for the given Gaussian distribution"""
        return dist.mean()


def thompson_sampling(dist, probs, values):
    if isinstance(dist, DiscreteDist):
        """Thompson sampling for the given discrete distribution."""
        return np.random.choice(values, p=probs)
    elif isinstance(dist, GaussDist):
        """Thompson sampling for the given Gaussian distribution"""
        return np.random.normal(loc=dist.mean(), scale=dist.std())
