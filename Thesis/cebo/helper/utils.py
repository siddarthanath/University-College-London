"""
This file stores utility functions, which will be used by other files.
"""
# -------------------------------------------------------------------------------------------------------------------- #

# Standard Library

# Third Party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Private Party
from cebo.helper.metrics import mean_squared_error
from cebo.helper.distmodel import DiscreteDist, GaussDist


# -------------------------------------------------------------------------------------------------------------------- #


def create_dataset(path, num_occurrences_low, num_occurrences_high, temps, num_smiles):
    data = pd.read_csv(path)
    data = data.dropna()
    data = data.drop_duplicates().reset_index(drop=True)
    data.rename(columns={"T,K": "Temperature"}, inplace=True)
    data = data.sort_values(by="SMILES")
    # Shrink dataset
    main_data = pd.DataFrame(columns=["SMILES"] + list(data["Temperature"].unique()))
    for smile in data["SMILES"].unique():
        sub_result = data[data["SMILES"] == smile]
        sub_temp = {"SMILES": smile}
        sub_temp.update(dict(sub_result["Temperature"].value_counts()))
        for temp in list(main_data.columns):
            if temp not in sub_temp.keys():
                sub_temp[temp] = 0
        main_data = pd.concat(
            (pd.DataFrame([sub_temp], columns=list(main_data.columns)), main_data)
        )
    sub_data = main_data[["SMILES"] + temps]
    mask = (sub_data.iloc[:, 1:] > num_occurrences_low) & (
        sub_data.iloc[:, 1:] < num_occurrences_high
    )
    mask = mask.all(axis=1)
    refined_data = sub_data[mask]
    refined_data = refined_data[refined_data.iloc[:, 1:].eq(5).all(axis=1)][:num_smiles]
    combined_data = data.merge(refined_data["SMILES"], on="SMILES")
    combined_data = combined_data[combined_data["Temperature"].isin(temps)]
    # Final dataframe
    combined_data.rename(columns={"SMILES_Solvent": "SMILES Solvent"}, inplace=True)
    combined_df = combined_data[
        ["SMILES", "Temperature", "SMILES Solvent", "Solubility"]
    ].reset_index(drop=True)
    return combined_df


def ablation_mse_results(full_results):
    # Store MSE results
    mse_results = {}
    for key, item in full_results.items():
        # Results
        sub_result = item
        # Store results for each iterations
        sub_experiment_results = {}
        for i in range(len(sub_result)):
            sub_experiment_mse_results = []
            sub_experiment = sub_result[i]
            # Loop through each iteration and calculate MSE
            for j in range(len(sub_experiment)):
                y_true = sub_experiment[j]["True"]
                y_pred = sub_experiment[j]["Predictions"]
                # Fill in empty or nan predictions - additionally, clip outliers as this will skew the MSE
                for k in range(len(y_pred)):
                    if np.isnan(y_pred[k]):
                        y_pred[k] = y_true[k]
                    elif np.abs(y_true[k] - y_pred[k]) > 100000:
                        y_pred[k] = 0
                    else:
                        y_pred[k] = y_pred[k]
                y_true = [x for x, y in zip(y_true, y_pred) if y != 0]
                y_pred = [y for y in y_pred if y != 0]
                # Calculate MSE
                mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
                # Store results
                results = {
                    "Iteration": j,
                    "T": sub_experiment[j]["T"],
                    "k": sub_experiment[j]["k"],
                    "Train": sub_experiment[j]["Train"],
                    "Test": sub_experiment[j]["Test"],
                    "Model": sub_experiment[j]["Model"],
                    "MSE": mse,
                }
                sub_experiment_mse_results.append(results)
            # Put everything into one dictionary
            final_mse_results = {
                "T": sub_experiment[j]["T"],
                "k": sub_experiment[j]["k"],
                "Train": sub_experiment[j]["Train"],
                "Test": sub_experiment[j]["Test"],
                "Model": sub_experiment[j]["Model"],
                "MSE": [experiment["MSE"] for experiment in sub_experiment_mse_results],
            }
            sub_experiment_results[i] = final_mse_results
        # Store results
        mse_results[key] = sub_experiment_results
        # Extract data for plotting
        full_results_thesis = []
        i = 0
        for _, data in mse_results.items():
            # Use log to scale results - since MSE values are above 1, this will not affect anything
            mse_data = [item["MSE"] for _, item in data.items()]
            T, k = data[i]["T"], data[i]["k"]
            # Create a figure and axis
            full_results_thesis.append(
                [
                    f"T = {T} | k = {k} | Result = {np.round(np.log(np.mean(value)), 3)}±{np.round(np.log(np.std(value)), 3)}"
                    for value in mse_data
                ]
            )
            i += 1
        return full_results_thesis


def process_bo_vs_cbo_results(results, selector, component):
    df = pd.DataFrame(columns=["Strategy", "Method", "Selection", f"{component}"])
    i = 0
    for method, method_data in results.items():
        # Iterate through the second-level dictionary
        for lift_type, lift_data in method_data.items():
            # Iterate through the 'Optimal Point with MMR' and 'Optimal Point without MMR' entries
            for j, (mmr_type, mmr_data) in enumerate(lift_data.items()):
                component_values = [
                    point[component] for point in results[method][lift_type][mmr_type]
                ]
                if len(component_values) != 0:
                    df.loc[i] = {
                        "Strategy": method,
                        "Method": lift_type,
                        "Selection": selector[i],
                        f"{component}": component_values,
                    }
                    i += 1
    return df


def plot_component_lists(component_lists, label, avg_label):
    assert is_list_of_floats_or_ints(component_lists)
    plt.figure(figsize=(10, 6))
    # Plot individual regret lists (slightly faded)
    for i, regret_list in enumerate(component_lists):
        plt.plot(np.cumsum(regret_list), alpha=0.5, label=f"{label}")
    # Calculate the average regret list
    avg_cum_regret = np.array(component_lists).mean(axis=0).cumsum()
    # Plot the average regret list (bold)
    plt.plot(
        avg_cum_regret, label=avg_label, linewidth=2.5, linestyle="--", color="black"
    )
    # Add labels and legend
    plt.xlabel("Iteration")
    plt.ylabel("Cumulative Regret")
    plt.title("Regret Over Iterations")
    plt.legend()


def is_list_of_floats_or_ints(lst):
    for inner_lst in lst:
        if not isinstance(inner_lst, list):
            return False
        for item in inner_lst:
            if not isinstance(item, (float, int)):
                return False
    return True


def combine(s, l):
    return s**l - (s - 1) ** (l)


def prob(s, l, n):
    return combine(s, l) * ((1 / n) ** l)


def expected_value_p(l, n):
    E = [s * prob(s, l, n) for s in range(1, 100 + 1)]
    return sum(E)


def expected_value_q(l, n, data):
    quants = [data.quantile(i / 100) for i in range(100 + 1)]
    # E = [(quants[s-1]) * prob(s, l, n) for s in range(1,100+1)]
    E = [((quants[s - 1] + quants[s]) / 2) * prob(s, l, n) for s in range(1, 100 + 1)]
    return sum(E)


def make_dd(values, probs):
    dd = DiscreteDist(values, probs)
    if len(dd) == 1:
        return GaussDist(dd.mean(), None)
    return dd
