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

# Private Party
from ..main import run_tell_predict
from ..utils import ablation_mse_results

# -------------------------------------------------------------------------------------------------------------------- #
os.environ["OPENAI_API_KEY"] = ""


@pytest.mark.parametrize("T_list, k_list, train_num_list, test_num_list, models_list",
                         [[0.7, 5, [5, 25, 45], [10], ["curie", "davinci"]]]
                         )
def test_tell_predict(T_list, k_list, train_num_list, test_num_list, models_list):
    # Clean and store data (NOTE: This should eventually be made into a fixture)
    aqsoldb_df = pd.read_csv("../cebo/data/aqsoldb.csv")
    aqsoldb_df = aqsoldb_df.dropna()
    aqsoldb_df = aqsoldb_df.drop_duplicates().reset_index(drop=True)
    aqsoldb_df.rename(columns={'Name': 'Compound ID'}, inplace=True)
    aqsoldb_df = aqsoldb_df.drop(["ID"], axis=1)
    aqsoldb_df = aqsoldb_df[aqsoldb_df["Compound ID"].str.len() < 15].reset_index(drop=True)
    aqsoldb_df = aqsoldb_df.sample(n=1000, random_state=42)
    # Results
    ablation_results = run_tell_predict(data=aqsoldb_df, T_list=T_list, k_list=k_list, train_num_list=train_num_list,
                                     test_num_list=test_num_list, models_list=models_list)
    full_results = {"BO-LIFT": ablation_results[0],
                    "BO-LIFT+DOMAIN": ablation_results[1],
                    "CEBO-LIFT+NO_DOMAIN": ablation_results[2],
                    "CEBO-LIFT+DOMAIN+FEATURE_1": ablation_results[3],
                    "CEBO-LIFT+DOMAIN+FEATURE_2": ablation_results[4],
                    }
    ablation_mse = ablation_mse_results(full_results=full_results)
