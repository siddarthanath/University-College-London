"""
This file tests the following experiments:
1. Contextual Bayesian Optimisation.
"""
# -------------------------------------------------------------------------------------------------------------------- #

# Standard Library
import os
import matplotlib.pyplot as plt

# Third Party
import pandas as pd
import pytest

# Private Party
from cebo.experiments.contextual_bo import run_bo_vs_c_bo
from cebo.helper.utils import (
    create_dataset,
    plot_component_lists,
    process_bo_vs_cbo_results,
)

# -------------------------------------------------------------------------------------------------------------------- #
os.environ["OPENAI_API_KEY"] = ""


@pytest.mark.parametrize("N, M, num_train, models_list", [20, 5, [10], ["curie"]])
def test_bo_vs_cbo(N, M, num_train, models_list):
    data = create_dataset(
        path="../cebo/data/bigsoldb.csv",
        num_occurrences_low=4,
        num_occurrences_high=6,
        temps=[313.15],
        num_smiles=2,
    )
    bo_vs_cbo_results = run_bo_vs_c_bo(
        data=data, N=N, M=M, num_train=num_train, models_list=models_list
    )
    # Obtain simplified results table
    results = bo_vs_cbo_results["upper_confidence_bound"]
    selector = ["with MMR", "without MMR"]
    component = "Regret"
    full_df = pd.DataFrame(columns=["Strategy", "Method", "Selection", f"{component}"])
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
