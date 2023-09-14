"""
This file automatically runs either the Tell-Predict or Contextual Bayesian Optimisation experiment.
"""
# -------------------------------------------------------------------------------------------------------------------- #

# Standard Library
import os
import matplotlib.pyplot as plt

# Third Party
import pandas as pd

# Private
from ..cebo.helper.utils import (
    create_dataset,
    process_bo_vs_cbo_results,
    plot_component_lists,
)
from ..cebo.experiments.contextual_bo import run_bo_vs_c_bo


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == "__main__":
    experiment_type = str(
        input(
            "Please enter either 'Tell-Predict' or 'Contextual BO' for your chosen experiment: \n"
        )
    )
    if experiment_type == "Contextual BO":
        # Install OpenAI API Key
        os.environ["OPENAI_API_KEY"] = str(
            input("Please input your OpenAI API Key: \n")
        )
        # Create data inputs
        data_input = input(
            "Please indicate the following, separated by a comma: \n"
            "Filepath of main data, lowest number of occurrences of a datapoint, "
            "highest number of occurrences of a datapoint, list of temperatures and number of distinct "
            "compounds: \n"
        )
        data_values = data_input.split(",")
        filepath = data_values[0].strip()
        lowest_occurrences = int(data_values[1].strip())
        highest_occurrences = int(data_values[2].strip())
        temperature_list = data_values[3]
        num_distinct_compounds = int(data_values[4].strip())
        # Create dataset
        data = create_dataset(
            path=filepath,
            num_occurrences_low=lowest_occurrences,
            num_occurrences_high=highest_occurrences,
            temps=temperature_list,
            num_smiles=num_distinct_compounds,
        )
        # Create results input
        result_input = input(
            "Please indicate the following, separated by a comma: \n"
            "Number of BO iterations, number of repeats per experiment, number of training points and "
            "a list of OpenAI model names: \n"
        )
        result_values = result_input.split(",")
        N = int(result_values[0].strip())
        M = int(result_values[1].strip())
        num_train = result_values[2].strip().split()
        models_list = result_values[3]
        # Run experiment
        bo_vs_cbo_results = run_bo_vs_c_bo(
            data=data, N=N, M=M, num_train=num_train, models_list=models_list
        )
        # Obtain simplified results table
        results = bo_vs_cbo_results["upper_confidence_bound"]
        selector = ["with MMR", "without MMR"]
        component = "Regret"
        full_df = pd.DataFrame(
            columns=["Strategy", "Method", "Selection", f"{component}"]
        )
        for result in results:
            df = process_bo_vs_cbo_results(
                results=result, selector=selector, component=component
            )
            full_df = pd.concat((df, full_df))
        full_df = full_df.reset_index(drop=True)
        full_grouped = (
            full_df.groupby(["Strategy", "Method", "Selection"])[f"{component}"]
            .apply(list)
            .reset_index()
        )
        # Plot BO vs C-BO Cumulative Regret
        for index, row in full_grouped.iterrows():
            strategy = row["Strategy"]
            method = row["Method"]
            selection = row["Selection"]
            component_lists = row["Regret"]
            label = f"{strategy}-{method}-{selection}"
            avg_label = f"Average - {label}"
            plot_component_lists(component_lists, label, avg_label)
            plt.show()
    else:
        print("Tell-Predict has not been configured yet.")
        pass
