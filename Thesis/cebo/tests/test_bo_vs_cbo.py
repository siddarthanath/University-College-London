"""
This file tests the following experiments:
1. Automatic Feature Engineering (for prompt design).
"""
# -------------------------------------------------------------------------------------------------------------------- #

# Standard Library
import os

# Third Party
import pandas as pd
import pytest
import matplotlib.pyplot as plt

# Private Party
from ..main import run_bo_vs_c_bo
from ..utils import process_bo_vs_cbo_results, plot_component_lists

# -------------------------------------------------------------------------------------------------------------------- #
os.environ["OPENAI_API_KEY"] = ""


@pytest.mark.parametrize("N, M, num_train, models_list",
                         [[15, 5, 10, ["curie"]]]
                         )
def test_bo_vs_cbo(N, M, num_train, models_list):
    # Call data
    data = pd.read_csv("/Users/siddarthanath/Documents/University-College-London/Thesis/cebo/data/bo_vs_cbo_df_1.csv")
    # Results
    bo_vs_cbo_results = run_bo_vs_c_bo(data=data, N=N, M=M, num_train=num_train, models_list=models_list)
    # Obtain simplified results table
    results = bo_vs_cbo_results["upper_confidence_bound"]
    strategies = ["BO", "C-BO"]
    methods = ["BO-LIFT", "CEBO-LIFT"]
    selector = ["with MMR", "without MMR"]
    component = "Regret"
    full_df = pd.DataFrame(columns=["Strategy", "Method", "Selection", f"{component}"])
    for result in results:
        df = process_bo_vs_cbo_results(results=result, methods=methods, selector=selector, strategies=strategies,
                                       component=component)
        full_df = pd.concat((df, full_df))
    full_df = full_df.reset_index(drop=True)
    full_grouped = full_df.groupby(['Strategy', 'Method', 'Selection'])[f'{component}'].apply(list).reset_index()
    # Plot BO vs C-BO Cumulative Regret
    for index, row in full_grouped.iterrows():
        strategy = row['Strategy']
        method = row['Method']
        selection = row['Selection']
        component_lists = row['Regret']
        label = f'{strategy}-{method}-{selection}'
        avg_label = f'Average - {label}'
        plot_component_lists(component_lists, label, avg_label)
        plt.show()
