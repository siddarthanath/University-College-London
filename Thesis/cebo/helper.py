"""
This file stores helper functions which are used within the general interface.
"""
# -------------------------------------------------------------------------------------------------------------------- #

# Standard Library
from typing import *
import random

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
from .llm import DiscreteDist, GaussDist


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


################################################################################################

def find_target(data, x_start, target):
    matching_index = (data.iloc[:, :-1] == x_start).all(axis=1)
    pos = matching_index[matching_index].index[0]
    y_start = data.loc[pos][target]
    return y_start


def find_experiment(data, result, context, t):
    result_summary = {key: value for key, value in result[0][0].items() if key != "Temperature"}
    sub_results = data[(data[list(result_summary)] == pd.Series(result_summary)).all(axis=1)]
    return data.loc[abs(sub_results[context] - t).idxmin()].to_dict()


################################################################################################
def process_frameworks(frameworks, data, target, context=None):
    # Check if context is None and raise ValueError if it's required for C-BO
    if context is None and "C-BO" in frameworks:
        raise ValueError("Context is required for C-BO.")
    # Extract relevant columns from data
    columns = ["SMILES", "SMILES Solvent"]
    # Tell
    for key_1, item_1 in frameworks.items():
        # Check for BO or C-BO
        if key_1 != "BO":
            columns.append(context)
        # Obtain example
        example = data[columns]
        # Tell
        for key_2, item_2 in item_1.items():
            if key_2 != "BO-LIFT":
                example[target] = data[target]
                example_columns = example.to_dict()
                [model.tell(example_columns) for model in item_2]
            else:
                example_columns = list(example)
                [model.tell(example_columns, data[target]) for model in item_2]


def process_optimal_points(frameworks, optimal_points_dict, context, target, selector):
    for key_1, item_1 in frameworks.items():
        for key_2, item_2 in item_1.items():
            for i, model in enumerate(item_2):
                opt_point = optimal_points_dict[key_1][key_2][f"Optimal Point {selector[i % 2]}"][0]
                opt_point_copy = opt_point.copy()
                if key_2 == "BO-LIFT":
                    try:
                        model.tell([opt_point_copy["SMILES"], opt_point_copy["SMILES Solvent"],
                                    opt_point_copy[context]], opt_point[target])
                    except BaseException as e:
                        opt_point_copy.pop(target)
                        model.tell([opt_point_copy["SMILES"], opt_point_copy["SMILES Solvent"]], opt_point[target])
                else:
                    pass


def generate_regret_structure(frameworks, f_t_max, y_start, x, results, selector):
    for key_1, item_1 in frameworks.items():
        for key_2, item_2 in item_1.items():
            for i, _ in enumerate(item_2):
                results[key_1][key_2][f"Optimal Point {selector[i % 2]}"].append({"Regret": f_t_max - y_start,
                                                                                  "Parameter": x,
                                                                                  "Temperature": x["Temperature"]})


def generate_optimising_point_structure(frameworks, pools, aq, methods, strategies, selector):
    results = {strategy: {method: {f"Optimal Point {selector[0]}": [], f"Optimal Point {selector[1]}": []} for method in
                          methods} for strategy in strategies}
    for key_1, item_1 in frameworks.items():
        for key_2, item_2 in item_1.items():
            for i, model in enumerate(item_2):
                results[key_1][key_2][f"Optimal Point {selector[i % 2]}"] = model.ask(pools[key_1], aq_fxn=aq,
                                                                                      _lambda=1.0)
    return results


def process_experiments(data, results, context, t, methods, strategies, selector):
    opt_points_df = []
    opt_points_dict = {
        strategy: {method: {f"Optimal Point {selector[0]}": [], f"Optimal Point {selector[1]}": []} for method in
                   methods} for strategy in strategies}
    for method, method_data in results.items():
        # Iterate through the second-level dictionary
        for lift_type, lift_data in method_data.items():
            # Iterate through the 'Optimal Point with MMR' and 'Optimal Point without MMR' entries
            for mmr_type, mmr_data in lift_data.items():
                if len(mmr_data) != 0:
                    opt_points_df.append(find_experiment(data, mmr_data, context, t))
                    opt_points_dict[method][lift_type][mmr_type].append(find_experiment(data, mmr_data, context, t))
    return pd.DataFrame(opt_points_df), opt_points_dict


def update_regret(current_results, new_results, f_t_max):
    # Update target values
    for method, method_data in current_results.items():
        # Iterate through the second-level dictionary
        for lift_type, lift_data in method_data.items():
            # Iterate through the 'Optimal Point with MMR' and 'Optimal Point without MMR' entries
            for mmr_type, mmr_data in lift_data.items():
                if len(mmr_data) != 0:
                    current_results[method][lift_type][mmr_type].append(
                        {"Regret": f_t_max - new_results[method][lift_type][mmr_type][0]["Solubility"],
                         "Parameter": new_results[method][lift_type][mmr_type][0],
                         "Temperature": new_results[method][lift_type][mmr_type][0]["Temperature"]})


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def run_experiment_cebo_lift_main(frameworks, data, indexes,
                                  context, target,
                                  N=10, initial_train=1, aq="random", start_index=0):
    # Acquisition function random
    if aq == 'random_mean':
        return [(i, expected_value_q(i, 100, data[target])) for i in range(1, N + 1)]
    # Tell
    for i in indexes[:initial_train]:
        process_frameworks(frameworks=frameworks, data=data.iloc[i, :], target=target, context=context)
    # Create pool
    pools = {"BO": [data.iloc[i, :][["SMILES", "SMILES Solvent"]].to_dict() for i in indexes],
             "C-BO": [data.iloc[i, :][["SMILES", "SMILES Solvent"] + [context]].to_dict() for i in indexes]}
    # Start point
    x_start = pools["C-BO"][start_index]
    # Obtain function output of start point
    y_start = find_target(data=data, x_start=x_start, target=target)
    x_start_copy = x_start.copy()
    x_start_copy.update({f"{target}": y_start})
    # Tell
    process_frameworks(frameworks=frameworks, data=pd.Series(x_start_copy), target=target, context=context)
    # Store regret
    strategies = ["BO", "C-BO"]
    methods = ["BO-LIFT", "CEBO-LIFT"]
    selector = ["with MMR", "without MMR"]
    f_t_max = data[data["Temperature"] == x_start["Temperature"]]["Solubility"].max()
    regret_results = {
        strategy: {method: {f"Optimal Point {selector[0]}": [], f"Optimal Point {selector[1]}": []} for method in
                   methods} for strategy in strategies}
    generate_regret_structure(frameworks, f_t_max, y_start, x_start_copy, regret_results, selector)
    # Initialise Bayesian Optimisation (BO) and Contextual Bayesian Optimisation (C-BO)
    for i in range(1, N):
        # Uniformly sample t ~ T (from pool)
        t = random.choice(data[context].tolist())
        # Remask the temperature of the pool candidates
        for j, ele in enumerate(pools["C-BO"]):
            ele["Temperature"] = t
            pools["C-BO"][j] = ele
        # BO & C-BO -
        # NOTE: remove duplicates in pool when querying!!!
        bo_and_cbo_results = generate_optimising_point_structure(frameworks, pools, aq, methods, strategies, selector)
        # Match the temperature sampled with the closest temperature in the pool for BO and C-BO
        _, final_opt_points_dict = process_experiments(data, bo_and_cbo_results, context, t,
                                                                         methods, strategies, selector)
        # Tell
        process_optimal_points(frameworks=frameworks, optimal_points_dict=final_opt_points_dict, context=context,
                               target=target, selector=selector)
        # Calculate f(x_t^*, t_t)
        f_t_max = data[data[context] == t]["Solubility"].max()
        # Update regret results
        update_regret(regret_results, final_opt_points_dict, f_t_max)
    return regret_results
