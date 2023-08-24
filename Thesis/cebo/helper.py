"""
This file stores helper functions which are used within the general interface.
"""
# -------------------------------------------------------------------------------------------------------------------- #

# Standard Library
from typing import *
import random
import copy

# Third Party
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

# Private
from llm import DiscreteDist, GaussDist


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


def mse(y, pred):
    # return np.mean((y-pred)**2)
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


def combine(s, l):
    '''Number of combinations of l elements with max = s'''
    return (s ** l - (s - 1) ** (l))


def prob(s, l, n):
    '''Probability of getting a sample with max([x0,x1,...,xl]) = s where xi={0,n}'''
    return combine(s, l) * ((1 / n) ** l)


def expected_value_p(l, n):
    '''Expected value of max([x0,x1,...,xl]) where xi={0,n}'''
    E = [s * prob(s, l, n) for s in range(1, 100 + 1)]
    return sum(E)


def expected_value_q(l, n, data):
    '''Expected value of max([x0,x1,...,xl]) where xi={0,n}'''
    quants = [data.quantile(i / 100) for i in range(100 + 1)]
    # E = [(quants[s-1]) * prob(s, l, n) for s in range(1,100+1)]
    E = [((quants[s - 1] + quants[s]) / 2) * prob(s, l, n) for s in range(1, 100 + 1)]
    return sum(E)


def find_target(data, x_start, target):
    matching_index = (data.iloc[:, :-1] == x_start).all(axis=1)
    pos = matching_index[matching_index].index[0]
    y_start = data.loc[pos][target]
    return y_start


def find_experiment(data, result, context, t):
    result_summary = {key: value for key, value in result[0][0].items() if key != "Temperature"}
    sub_results = data[(data[list(result_summary)] == pd.Series(result_summary)).all(axis=1)]
    return data.loc[abs(sub_results[context] - t).idxmin()].to_dict()


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def run_experiment_cebo_lift(model_1, model_2, data, indexes,
                             context, target,
                             N=10, initial_train=1, aq="random", start_index=0):
    # Acquisition function random
    if aq == 'random_mean':
        return [(i, expected_value_q(i, 100, data[target])) for i in range(1, N + 1)]
    # Tell
    for i in indexes[:initial_train]:
        example_1 = data.iloc[i, :][["SMILES", "SMILES Solvent"] + [target]]
        model_1.tell(example_1.to_dict())
        example_2 = data.iloc[i, :][["SMILES", "SMILES Solvent"] + [context] + [target]]
        model_2.tell(example_2.to_dict())
    # Create pool
    bo_pool = [data.iloc[i, :][["SMILES", "SMILES Solvent"]].to_dict() for i in indexes]
    cbo_pool = [data.iloc[i, :][["SMILES", "SMILES Solvent"] + [context]].to_dict() for i in indexes]
    # Start point
    x_start = cbo_pool[start_index]
    # Obtain function output of start point
    y_start = find_target(data=data, x_start=x_start, target=target)
    # Tell
    x_copy_bo = x_start.copy()
    x_copy_cbo = x_start.copy()
    x_copy_bo.pop("Temperature")
    x_copy_bo.update({f"{target}": y_start})
    x_copy_cbo.update({f"{target}": y_start})
    model_1.tell(x_copy_bo)
    model_2.tell(x_copy_cbo)
    # Store regret
    f_t_max = data[data["Temperature"] == x_start["Temperature"]]["Solubility"].max()
    regret_bo_t = {0: {"Regret": f_t_max - y_start, "Parameter": x_copy_cbo, "Temperature": x_start["Temperature"]}}
    regret_cbo_t = {0: {"Regret": f_t_max - y_start, "Parameter": x_copy_cbo, "Temperature": x_start["Temperature"]}}
    # Initialise Bayesian Optimisation (BO) and Contextual Bayesian Optimisation (C-BO)
    for i in range(1, N):
        # Uniformly sample t ~ T (from pool)
        t = random.choice(data["Temperature"].tolist())
        # Remask the temperature of the pool candidates
        for j, ele in enumerate(cbo_pool):
            ele["Temperature"] = t
            cbo_pool[j] = ele
        # BO
        result_bo = model_1.ask(bo_pool, aq_fxn=aq, _lambda=1.0)
        # C-BO
        result_cbo = model_2.ask(cbo_pool, aq_fxn=aq, _lambda=1.0)
        # Match the temperature sampled with the closest temperature in the pool for BO and C-BO
        bo_x_t = find_experiment(data=data, result=result_bo, context=context, t=t)
        cbo_x_t = find_experiment(data=data, result=result_cbo, context=context, t=t)
        # Tell
        model_1.tell(pd.Series(bo_x_t).drop(["Temperature"]).to_dict())
        model_2.tell(pd.Series(cbo_x_t).to_dict())
        # Calculate f(x_t, t_t)
        y_bo = bo_x_t["Solubility"]
        y_cbo = cbo_x_t["Solubility"]
        # Calculate f(x_t^*, t_t)
        f_t_max = data[data["Temperature"] == t]["Solubility"].max()
        # Calculate regret i.e. f(x_t^*, t_t)-f(x_t, t_t)
        regret_bo_t[i] = {"Regret": f_t_max - y_bo, "Parameter": bo_x_t, "Temperature": t}
        regret_cbo_t[i] = {"Regret": f_t_max - y_cbo, "Parameter": cbo_x_t, "Temperature": t}
    return regret_bo_t, regret_cbo_t



@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def run_experiment_cebo_lift_other(model_1, model_2, data, indexes,
                             context, target,
                             N=10, initial_train=1, aq="random", start_index=0):
    # Acquisition function random
    if aq == 'random_mean':
        return [(i, expected_value_q(i, 100, data[target])) for i in range(1, N + 1)]
    # Tell
    for i in indexes[:initial_train]:
        example_1 = data.iloc[i, :][["SMILES", "SMILES Solvent"] + [target]]
        model_1.tell([example_1["SMILES"]]+[example_1["SMILES Solvent"]], example_1[target])
        example_2 = data.iloc[i, :][["SMILES", "SMILES Solvent"] + [context] + [target]]
        model_2.tell(example_2.to_dict())
    # Create pool
    bo_pool = [data.iloc[i, :][["SMILES", "SMILES Solvent"]].to_dict() for i in indexes]
    cbo_pool = [data.iloc[i, :][["SMILES", "SMILES Solvent"] + [context]].to_dict() for i in indexes]
    # Start point
    x_start = cbo_pool[start_index]
    # Obtain function output of start point
    y_start = find_target(data=data, x_start=x_start, target=target)
    # Tell
    x_copy_bo = x_start.copy()
    x_copy_cbo = x_start.copy()
    x_copy_bo.pop("Temperature")
    x_copy_bo.update({f"{target}": y_start})
    x_copy_cbo.update({f"{target}": y_start})
    model_1.tell([x_copy_bo["SMILES"]]+[x_copy_bo["SMILES Solvent"]], x_copy_bo[target])
    model_2.tell(x_copy_cbo)
    # Store regret
    f_t_max = data[data["Temperature"] == x_start["Temperature"]]["Solubility"].max()
    regret_bo_t = {0: {"Regret": f_t_max - y_start, "Parameter": x_copy_cbo, "Temperature": x_start["Temperature"]}}
    regret_cbo_t = {0: {"Regret": f_t_max - y_start, "Parameter": x_copy_cbo, "Temperature": x_start["Temperature"]}}
    # Initialise Bayesian Optimisation (BO) and Contextual Bayesian Optimisation (C-BO)
    for i in range(1, N):
        # Uniformly sample t ~ T (from pool)
        t = random.choice(data["Temperature"].tolist())
        # Remask the temperature of the pool candidates
        for j, ele in enumerate(cbo_pool):
            ele["Temperature"] = t
            cbo_pool[j] = ele
        # BO
        result_bo = model_1.ask(bo_pool, aq_fxn=aq, _lambda=1.0)
        # C-BO
        result_cbo = model_2.ask(cbo_pool, aq_fxn=aq, _lambda=1.0)
        # Match the temperature sampled with the closest temperature in the pool for BO and C-BO
        bo_x_t = find_experiment(data=data, result=result_bo, context=context, t=t)
        cbo_x_t = find_experiment(data=data, result=result_cbo, context=context, t=t)
        # Tell
        bo_x_t_final = pd.Series(bo_x_t).drop(["Temperature"]).to_dict()
        model_1.tell([bo_x_t_final["SMILES"]]+[bo_x_t_final["SMILES Solvent"]], bo_x_t_final[target])
        model_2.tell(pd.Series(cbo_x_t).to_dict())
        # Calculate f(x_t, t_t)
        y_bo = bo_x_t["Solubility"]
        y_cbo = cbo_x_t["Solubility"]
        # Calculate f(x_t^*, t_t)
        f_t_max = data[data["Temperature"] == t]["Solubility"].max()
        # Calculate regret i.e. f(x_t^*, t_t)-f(x_t, t_t)
        regret_bo_t[i] = {"Regret": f_t_max - y_bo, "Parameter": bo_x_t, "Temperature": t}
        regret_cbo_t[i] = {"Regret": f_t_max - y_cbo, "Parameter": cbo_x_t, "Temperature": t}
    return regret_bo_t, regret_cbo_t

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def run_experiment_cebo_lift_other_2(model_1, model_2, data, indexes,
                             context, target,
                             N=10, initial_train=1, aq="random", start_index=0):
    # Acquisition function random
    if aq == 'random_mean':
        return [(i, expected_value_q(i, 100, data[target])) for i in range(1, N + 1)]
    # Tell
    for i in indexes[:initial_train]:
        example_1 = data.iloc[i, :][["SMILES", "SMILES Solvent"] + [target]]
        model_1.tell([example_1["SMILES"]]+[example_1["SMILES Solvent"]], example_1[target])
        example_2 = data.iloc[i, :][["SMILES", "SMILES Solvent"] + [context] + [target]]
        model_2.tell([example_2["SMILES"]]+[example_2["SMILES Solvent"]] + [example_2[context]], example_1[target])
    # Create pool
    bo_pool = [data.iloc[i, :][["SMILES", "SMILES Solvent"]].to_dict() for i in indexes]
    cbo_pool = [data.iloc[i, :][["SMILES", "SMILES Solvent"] + [context]].to_dict() for i in indexes]
    # Start point
    x_start = cbo_pool[start_index]
    # Obtain function output of start point
    y_start = find_target(data=data, x_start=x_start, target=target)
    # Tell
    x_copy_bo = x_start.copy()
    x_copy_cbo = x_start.copy()
    x_copy_bo.pop("Temperature")
    x_copy_bo.update({f"{target}": y_start})
    x_copy_cbo.update({f"{target}": y_start})
    model_1.tell([x_copy_bo["SMILES"]]+[x_copy_bo["SMILES Solvent"]], x_copy_bo[target])
    model_2.tell([x_copy_cbo["SMILES"]]+[x_copy_cbo["SMILES Solvent"]]+[x_copy_cbo[context]], x_copy_bo[target])
    # Store regret
    f_t_max = data[data["Temperature"] == x_start["Temperature"]]["Solubility"].max()
    regret_bo_t = {0: {"Regret": f_t_max - y_start, "Parameter": x_copy_cbo, "Temperature": x_start["Temperature"]}}
    regret_cbo_t = {0: {"Regret": f_t_max - y_start, "Parameter": x_copy_cbo, "Temperature": x_start["Temperature"]}}
    # Initialise Bayesian Optimisation (BO) and Contextual Bayesian Optimisation (C-BO)
    for i in range(1, N):
        # Uniformly sample t ~ T (from pool)
        t = random.choice(data[context].tolist())
        # Remask the temperature of the pool candidates
        for j, ele in enumerate(cbo_pool):
            ele["Temperature"] = t
            cbo_pool[j] = ele
        # BO
        result_bo = model_1.ask(bo_pool, aq_fxn=aq, _lambda=1.0)
        # C-BO
        result_cbo = model_2.ask(cbo_pool, aq_fxn=aq, _lambda=1.0)
        # Match the temperature sampled with the closest temperature in the pool for BO and C-BO
        bo_x_t = find_experiment(data=data, result=result_bo, context=context, t=t)
        cbo_x_t = find_experiment(data=data, result=result_cbo, context=context, t=t)
        # Tell
        bo_x_t_final = pd.Series(bo_x_t).drop(["Temperature"]).to_dict()
        model_1.tell([bo_x_t_final["SMILES"]]+[bo_x_t_final["SMILES Solvent"]], bo_x_t_final[target])
        model_2.tell([cbo_x_t["SMILES"]]+[cbo_x_t["SMILES Solvent"]] + [cbo_x_t[context]], cbo_x_t[target])
        # Calculate f(x_t, t_t)
        y_bo = bo_x_t["Solubility"]
        y_cbo = cbo_x_t["Solubility"]
        # Calculate f(x_t^*, t_t)
        f_t_max = data[data[context] == t]["Solubility"].max()
        # Calculate regret i.e. f(x_t^*, t_t)-f(x_t, t_t)
        regret_bo_t[i] = {"Regret": f_t_max - y_bo, "Parameter": bo_x_t, "Temperature": t}
        regret_cbo_t[i] = {"Regret": f_t_max - y_cbo, "Parameter": cbo_x_t, "Temperature": t}
    return regret_bo_t, regret_cbo_t