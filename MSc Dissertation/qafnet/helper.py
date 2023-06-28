"""
This file stores helper functions which are used within the general interface.
"""
# -------------------------------------------------------------------------------------------------------------------- #

# Standard Library
import re
from typing import *

# Third Party
import numpy as np
from scipy.stats import norm
from pydantic import create_model

# Private
from .llm_model import DiscreteDist, GaussDist

# -------------------------------------------------------------------------------------------------------------------- #

def sanitize_field_name(name: str) -> str:
    """
    This function cleans input names so that it can be recognised by the pydantic model.
    
    Args:
        name:
            The key word argument in the pydantic model.
    Returns:
        A clean string Python variable name output.
    
    """
    # Use regular expression to clean string name
    sanitized = name.strip().replace(' ', '_')
    sanitized = re.sub(r'\W|^(?=\d)', '', sanitized)
    if sanitized and sanitized[0].isdigit():
        raise ValueError(f"Invalid field name '{name}': After sanitization, field name cannot start with a digit.")
    return sanitized


def generate_model(context: Dict[str, type]):
    """
    This function creates the pydantic model.
    
    Args:
        context:  
            The context of the problem - they key is the input name and the value is the datatype.
    Returns:

    """
    # Create fields
    fields = {sanitize_field_name(k): (t, ...) for k, t in context.items()}
    return create_model('DynamicModel', **fields)


def expected_improvement(dist: Union[DiscreteDist, GaussDist], best: float):
    """
    Calculate expected improvement for the given discrete distribution.
    
    Args:
        dist:
            The distribution of the results.
        best:
            The input which has the best acquisition value associated with it.
    """
    if isinstance(dist, DiscreteDist):
        return expected_improvement_d(dist.probs, dist.values, best)
    elif isinstance(dist, GaussDist):
        return expected_improvement_g(dist.mean(), dist.std(), best)


def probability_of_improvement(dist, best):
    """Probability of improvement for the given discrete distribution"""
    if isinstance(dist, DiscreteDist):
        return probability_of_improvement_d(dist.probs, dist.values, best)
    elif isinstance(dist, GaussDist):
        return probability_of_improvement_g(dist.mean(), dist.std(), best)


def upper_confidence_bound(dist, best, _lambda):
    """Upper confidence bound for the given discrete distribution"""
    if isinstance(dist, DiscreteDist):
        return upper_confidence_bound_d(dist.probs, dist.values, best, _lambda)
    elif isinstance(dist, GaussDist):
        return upper_confidence_bound_g(dist.mean(), dist.std(), best, _lambda)


def greedy(dist, best):
    """Greedy selection (most likely point) for the given discrete distribution"""
    if isinstance(dist, DiscreteDist):
        return greedy_d(dist.probs, dist.values, best)
    elif isinstance(dist, GaussDist):
        return greedy_g(dist.mean(), dist.std(), best)


def expected_improvement_d(probs, values, best):
    """Expected improvement for the given discrete distribution"""
    ei = np.sum(np.maximum(values - best, 0) * probs)
    return ei


def probability_of_improvement_d(probs, values, best):
    """Probability of improvement for the given discrete distribution"""
    pi = np.sum(np.cast[float](values > best) * probs)
    return pi


def upper_confidence_bound_d(probs, values, best, _lambda):
    """Upper confidence bound for the given discrete distribution"""
    mu = np.sum(values * probs)
    sigma = np.sqrt(np.sum((values - mu) ** 2 * probs))
    return mu + _lambda * sigma


def greedy_d(probs, values, best):
    """Greedy selection (most likely point) for the given discrete distribution"""
    return values[np.argmax(probs)]


def expected_improvement_g(mean, std, best):
    """Expected improvement for the given Gaussian distribution"""
    z = (mean - best) / std
    ei = (mean - best) * norm.cdf(z) + std * norm.pdf(z)
    return ei


def probability_of_improvement_g(mean, std, best):
    """Probability of improvement for the given Gaussian distribution"""
    z = (mean - best) / std
    pi = norm.cdf(z)
    return pi


def upper_confidence_bound_g(mean, std, best, _lambda):
    """Upper confidence bound for the given Gaussian distribution"""
    return mean + _lambda * std


def greedy_g(mean, std, best):
    """Greedy selection (most likely point) for the given Gaussian distribution"""
    return mean
