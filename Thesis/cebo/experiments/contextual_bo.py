"""
This file carries out traditional and contextual Bayesian Optimisation (depending on the framework).
"""
# -------------------------------------------------------------------------------------------------------------------- #

# Standard Library
import random
import itertools

# Third Party
import numpy as np
import pandas as pd
import uncertainty_toolbox as uct
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

# Private
from cebo.models.cebo_lift import CEBOLIFT
from cebo.models.bo_lift import BOLIFT


# -------------------------------------------------------------------------------------------------------------------- #


def run_bo_vs_c_bo(data, N, M, num_train, models_list, _lambda):
    # Create model
    bayesOpts_results = {}
    # Loop through models
    for model, no_train, lambda_ in itertools.product(models_list, num_train, _lambda):
        # Parameters
        indexes = np.arange(0, data.shape[0])
        np.random.shuffle(indexes)
        print(
            f"Model = {model} | Number of training points = {no_train} | Lambda Value = {lambda_}"
        )
        regret_total_results = []
        starts = np.random.randint(0, len(indexes), M)
        for k in range(M):
            # BO-LIFT with MMR (BO)
            bo_lift_1 = BOLIFT(
                x_formatter=lambda x: f"SMILES {x[0]}, SMILES Solvent {x[1]}",
                y_name="solubility",
                y_formatter=lambda y: f"{y:.8f}",
                model=model,
                selector_k=10,
                temperature=0.75,
            )
            # BO-LIFT with MMR (BO)
            bo_lift_3 = BOLIFT(
                x_formatter=lambda x: f"SMILES {x[0]}, SMILES Solvent {x[1]}, Temperature {x[2]}",
                y_name="solubility",
                y_formatter=lambda y: f"{y:.8f}",
                model=model,
                selector_k=10,
                temperature=0.75,
            )
            # CEBO-LIFT with MMR (C-BO)
            cebo_lift_1 = CEBOLIFT(
                y_name="solubility",
                model=model,
                selector_k=10,
                temperature=0.75,
                domain="chemist",
                features=True,
            )
            framework_types = {
                "BO": {"BO-LIFT": [bo_lift_1]},
                "C-BO": {"CEBO-LIFT": [cebo_lift_1]},
            }
            regret_results = run_experiment_cebo_lift_main(
                frameworks=framework_types,
                data=data,
                indexes=indexes,
                context="Temperature",
                target="Solubility",
                N=N,
                aq_fxn="upper_confidence_bound",
                lambda_=lambda_,
                initial_train=no_train,
                start_index=starts[k],
            )
            regret_total_results.append(regret_results)
        # Store results
        bayesOpts_results[f"{model}/{no_train}/{lambda_}"] = regret_total_results
    # Return results
    return bayesOpts_results


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def run_experiment_cebo_lift_main(
        frameworks,
        data,
        indexes,
        context,
        target,
        aq_fxn,
        N=10,
        lambda_=1,
        initial_train=1,
        start_index=0,
):
    # Store keys
    strategies = ["BO", "C-BO"]
    methods = ["BO-LIFT", "CEBO-LIFT"]
    selector = ["with MMR", "without MMR"]
    # Tell
    [
        process_frameworks(
            frameworks=frameworks, data=data.iloc[i, :], target=target, context=context
        )
        for i in indexes[:initial_train]
    ]
    # Calibrate
    sub_pool = {
        "BO": [
            data.iloc[i, :][["SMILES", "SMILES Solvent"]].to_dict()
            for i in indexes[:initial_train]
        ],
        "C-BO": [
            data.iloc[i, :][["SMILES", "SMILES Solvent"] + [context]].to_dict()
            for i in indexes[:initial_train]
        ],
    }
    # y = [data.loc[i][target] for i in indexes[:initial_train]]
    # calibrate_models(frameworks, sub_pool, y)
    # # Create pool
    pools = {
        "BO": [
            data.iloc[i, :][["SMILES", "SMILES Solvent"]].to_dict() for i in indexes
        ],
        "C-BO": [
            data.iloc[i, :][["SMILES", "SMILES Solvent"] + [context]].to_dict()
            for i in indexes
        ],
    }
    # Start point
    x_start = pools["C-BO"][start_index]
    # Obtain function output of start point
    y_start = find_target(data=data, x_start=x_start, target=target)
    x_start_copy = x_start.copy()
    x_start_copy.update({f"{target}": y_start})
    # Tell
    if initial_train != 0:
        process_frameworks(
            frameworks=frameworks,
            data=pd.Series(x_start_copy),
            target=target,
            context=context,
        )
    # Store regret
    f_t_max = data[data["Temperature"] == x_start["Temperature"]]["Solubility"].max()
    regret_results = {
        strategy: {
            method: {
                f"Optimal Point {selector[0]}": [],
                f"Optimal Point {selector[1]}": [],
            }
            for method in methods
        }
        for strategy in strategies
    }
    # Calculate number of context
    num_context = len(data["Temperature"].unique())
    generate_regret_structure(
        frameworks,
        f_t_max,
        y_start,
        x_start_copy,
        regret_results,
        selector,
        context,
        num_context,
        t=-1
    )
    # Obtain best (for single context)
    best = {"BO": y_start,
            "C-BO": y_start}
    # Initialise Bayesian Optimisation (BO) and Contextual Bayesian Optimisation (C-BO)
    for i in range(1, N):
        # Uniformly sample t ~ T (from pool)
        t = random.choice(np.unique(data[context].tolist()))
        # Remask the temperature of the pool candidates
        for j, ele in enumerate(pools["C-BO"]):
            ele["Temperature"] = t
            pools["C-BO"][j] = ele
        # BO & C-BO
        bo_and_cbo_results = generate_optimising_point_structure(data, aq_fxn,
                                                                 frameworks, pools, methods, strategies, lambda_,
                                                                 selector
                                                                 )
        # Match the temperature sampled with the closest temperature in the pool for BO and C-BO
        _, final_opt_points_dict = process_experiments(
            data, bo_and_cbo_results, context, t, methods, strategies, selector
        )
        # Tell
        process_optimal_points(
            frameworks=frameworks,
            optimal_points_dict=final_opt_points_dict,
            context=context,
            target=target,
            selector=selector,
        )
        # Calculate f(x_t^*, t_t)
        f_t_max = data[data[context] == t]["Solubility"].max()
        # Update regret results
        update_regret(regret_results, final_opt_points_dict, f_t_max, num_context, best, t)
        # Remove candidate from pool
        if num_context < 2:
            # BO point
            bo_point = {key: value for key, value in
                        final_opt_points_dict["BO"]["BO-LIFT"]["Optimal Point with MMR"][0][0].items() if
                        key not in {"Solubility", "Temperature"}}
            pools["BO"].pop(pools["BO"].index(bo_point))
            # C-BO point
            cbo_point = {key: value for key, value in
                         final_opt_points_dict["C-BO"]["CEBO-LIFT"]["Optimal Point with MMR"][0][0].items() if
                         key not in {"Solubility"}}
            pools["C-BO"].pop(pools["C-BO"].index(cbo_point))
            # Check if maximum is reached and apply early stopping
            if best["BO"] == f_t_max and best["C-BO"] == f_t_max:
                break
    return regret_results


def find_target(data, x_start, target):
    matching_index = (data.iloc[:, :-1] == x_start).all(axis=1)
    pos = matching_index[matching_index].index[0]
    y_start = data.loc[pos][target]
    return y_start


def find_experiment(data, result, context, t):
    result_summary = {
        key: value for key, value in result[0].items() if key != "Temperature"
    }
    sub_results = data[
        (data[list(result_summary)] == pd.Series(result_summary)).all(axis=1)
    ]
    return data.loc[abs(sub_results[context] - t).idxmin()].to_dict(), result[1]


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


def calibrate_models(frameworks, pools, y):
    for key_1, item_1 in frameworks.items():
        for key_2, item_2 in item_1.items():
            for i, model in enumerate(item_2):
                pool = pools[key_1]
                unique_tuple = {tuple(sorted(d.items())) for d in pool}
                unique_pool = [dict(t) for t in unique_tuple]
                model = frameworks[key_1][key_2][i]
                model._calibration_factor = True
                pred = [model.predict(candidate) for candidate in unique_pool]
                # Calibrate models
                ymeans = np.array([yhi.mean() for yhi in pred])
                ystds = np.array([yhi.std() for yhi in pred])
                calibration_factor = uct.recalibration.optimize_recalibration_ratio(
                    ymeans, ystds, np.array(y), criterion="miscal"
                )
                model.set_calibration_factor(calibration_factor)


def process_optimal_points(frameworks, optimal_points_dict, context, target, selector):
    for key_1, item_1 in frameworks.items():
        for key_2, item_2 in item_1.items():
            for i, model in enumerate(item_2):
                opt_point = optimal_points_dict[key_1][key_2][
                    f"Optimal Point {selector[i % 2]}"
                ][0][0]
                opt_point_copy = opt_point.copy()
                if key_2 == "BO-LIFT":
                    try:
                        model.tell(
                            [
                                opt_point_copy["SMILES"],
                                opt_point_copy["SMILES Solvent"],
                                opt_point_copy[context],
                            ],
                            opt_point[target],
                        )
                    except BaseException as e:
                        opt_point_copy.pop(target)
                        model.tell(
                            [
                                opt_point_copy["SMILES"],
                                opt_point_copy["SMILES Solvent"],
                            ],
                            opt_point[target],
                        )
                else:
                    model.tell(opt_point_copy)


def generate_regret_structure(
        frameworks, f_t_max, y_start, x, results, selector, context, num_context, t
):
    for key_1, item_1 in frameworks.items():
        for key_2, item_2 in item_1.items():
            for i, _ in enumerate(item_2):
                if num_context > 1:
                    results[key_1][key_2][f"Optimal Point {selector[i % 2]}"].append(
                        {
                            "Regret": f_t_max - y_start,
                            "Parameter": x,
                            "Temperature": x[context],
                            "Acquisition Value": 0,
                            "Sampled Temperature": t,
                        }
                    )
                else:
                    results[key_1][key_2][f"Optimal Point {selector[i % 2]}"].append(
                        {
                            "Regret": 0,
                            "Parameter": x,
                            "Temperature": x[context],
                            "Acquisition Value": 0,
                            "Sampled Temperature": t,
                        }
                    )


def generate_optimising_point_structure(
        data, aq_fxn, frameworks, pools, methods, strategies, lambda_, selector
):
    results = {
        strategy: {
            method: {
                f"Optimal Point {selector[0]}": [],
                f"Optimal Point {selector[1]}": [],
            }
            for method in methods
        }
        for strategy in strategies
    }
    for key_1, item_1 in frameworks.items():
        for key_2, item_2 in item_1.items():
            for i, model in enumerate(item_2):
                pool = pools[key_1]
                unique_tuple = {tuple(sorted(d.items())) for d in pool}
                unique_pool = [dict(t) for t in unique_tuple]
                results[key_1][key_2][f"Optimal Point {selector[i % 2]}"] = model.ask(
                    unique_pool, aq_fxn, data, _lambda=lambda_
                )
    return results


def process_experiments(data, results, context, t, methods, strategies, selector):
    opt_points_df = []
    opt_points_dict = {
        strategy: {
            method: {
                f"Optimal Point {selector[0]}": [],
                f"Optimal Point {selector[1]}": [],
            }
            for method in methods
        }
        for strategy in strategies
    }
    for method, method_data in results.items():
        # Iterate through the second-level dictionary
        for lift_type, lift_data in method_data.items():
            # Iterate through the 'Optimal Point with MMR' and 'Optimal Point without MMR' entries
            for mmr_type, mmr_data in lift_data.items():
                if len(mmr_data) != 0:
                    opt_points_df.append(find_experiment(data, mmr_data, context, t))
                    opt_points_dict[method][lift_type][mmr_type].append(
                        find_experiment(data, mmr_data, context, t)
                    )
    return pd.DataFrame(opt_points_df), opt_points_dict


def update_regret(current_results, new_results, f_t_max, num_context, best, t):
    # Update target values
    for method, method_data in current_results.items():
        # Iterate through the second-level dictionary
        for lift_type, lift_data in method_data.items():
            # Iterate through the 'Optimal Point with MMR' and 'Optimal Point without MMR' entries
            for mmr_type, mmr_data in lift_data.items():
                if len(mmr_data) != 0:
                    if num_context > 1:
                        current_results[method][lift_type][mmr_type].append(
                            {
                                "Regret": f_t_max
                                          - new_results[method][lift_type][mmr_type][0][0]["Solubility"],
                                "Parameter": new_results[method][lift_type][mmr_type][0][0],
                                "Temperature": new_results[method][lift_type][mmr_type][0][0][
                                    "Temperature"
                                ],
                                "Acquisition Value": new_results[method][lift_type][mmr_type][0][1],
                                "Sampled Temperature": t,
                            }
                        )
                    else:
                        current_results[method][lift_type][mmr_type].append(
                            {
                                "Regret": max(best[method], new_results[method][lift_type][mmr_type][0][0][
                                    "Solubility"
                                ]),
                                "Parameter": new_results[method][lift_type][mmr_type][0][0],
                                "Temperature": new_results[method][lift_type][mmr_type][0][0][
                                    "Temperature"
                                ],
                                "Acquisition Value": new_results[method][lift_type][mmr_type][0][1],
                                "Sampled Temperature": t,
                            }
                        )
                        # Update best
                        best[method] = max(best[method], new_results[method][lift_type][mmr_type][0][0][
                            "Solubility"
                        ])
