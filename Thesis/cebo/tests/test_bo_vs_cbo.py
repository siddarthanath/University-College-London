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

# Private Party
from cebo.experiments.contextual_bo import run_bo_vs_c_bo
from cebo.helper.utils import (
    create_dataset,
    plot_component_lists,
    process_bo_vs_cbo_results,
)

# -------------------------------------------------------------------------------------------------------------------- #
os.environ["OPENAI_API_KEY"] = "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"


def test_bo_vs_cbo_one_temperature():
    N = 25
    M = 5
    num_train = [5, 15, 25]
    models_list = ["gpt-3.5-turbo"]
    _lambda = [2, 4, 6]
    data = pd.read_csv(
        "/Users/siddarthanath/Documents/University-College-London/Thesis/cebo/tests/bo_vs_cbo_1.csv"
    )
    bo_vs_cbo_results = run_bo_vs_c_bo(
        data=data,
        N=N,
        M=M,
        num_train=num_train,
        models_list=models_list,
        _lambda=_lambda,
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
