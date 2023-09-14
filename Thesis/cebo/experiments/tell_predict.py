"""
This file carries out the Tell-Predict experiment.
"""
# -------------------------------------------------------------------------------------------------------------------- #

# Standard Library
import itertools

# Third Party
import numpy as np

# Private
from cebo.models.cebo_lift import CEBOLIFT
from cebo.models.bo_lift import BOLIFT


# -------------------------------------------------------------------------------------------------------------------- #


def run_tell_predict(data, T_list, k_list, train_num_list, test_num_list, models_list):
    # Store results
    bo_lift_results = []
    cebo_lift_results_1 = []
    cebo_lift_results_2 = []
    cebo_lift_results_3 = []
    cebo_lift_results_4 = []
    cebo_lift_results_5 = []
    # Loop
    for T, k, num_train, num_test, model in itertools.product(
        T_list, k_list, train_num_list, test_num_list, models_list
    ):
        bo_lift_result = []
        cebo_lift_result_1 = []
        cebo_lift_result_2 = []
        cebo_lift_result_3 = []
        cebo_lift_result_4 = []
        cebo_lift_result_5 = []
        for i in range(10):
            # Create data
            shuffled_df = data.sample(frac=1, random_state=42)
            train_df = shuffled_df.iloc[:num_train]
            test_df = shuffled_df.iloc[num_train:].head(num_test)
            # Create the model object
            bo_lift = BOLIFT(
                x_formatter=lambda x: f"compound id {x}",
                y_name="solubility",
                y_formatter=lambda y: f"{y:.6f}",
                model=model,
                selector_k=k,
                temperature=0.7,
            )
            cebo_lift_1 = BOLIFT(
                x_formatter=lambda x: f"compound id {x}",
                y_name="solubility",
                y_formatter=lambda y: f"{y:.6f}",
                model=model,
                selector_k=k,
                temperature=T,
                prefix=(
                    f"You are an expert chemist. "
                    "The following are correctly answered questions. "
                    "Each answer is numeric and ends with ###\n"
                ),
            )
            cebo_lift_2 = CEBOLIFT(
                y_name="solubility",
                model=model,
                selector_k=k,
                temperature=T,
                domain=None,
                features=True,
            )
            cebo_lift_3 = CEBOLIFT(
                y_name="solubility",
                model=model,
                selector_k=k,
                temperature=T,
                domain="chemist",
                features=True,
            )
            cebo_lift_4 = CEBOLIFT(
                y_name="solubility",
                model=model,
                selector_k=k,
                temperature=T,
                domain="chemist",
                features=True,
            )
            cebo_lift_5 = CEBOLIFT(
                y_name="solubility",
                model=model,
                selector_k=k,
                temperature=T,
                domain="chemist",
                features=True,
            )
            # Tell some points to the model
            for _, row in train_df.iterrows():
                bo_lift.tell(row["Compound ID"], row["Solubility"])
                cebo_lift_1.tell(row["Compound ID"], row["Solubility"])
                cebo_lift_2.tell(
                    row[["Compound ID", "MolLogP", "MolMR", "Solubility"]].to_dict()
                )
                cebo_lift_3.tell(
                    row[["Compound ID", "MolLogP", "MolMR", "Solubility"]].to_dict()
                )
                cebo_lift_4.tell(
                    row[["Compound ID", "Ocurrences", "SD", "Solubility"]].to_dict()
                )
                cebo_lift_5.tell(
                    row[["Compound ID", "Ocurrences", "SD", "Solubility"]].to_dict()
                )
            # Predict remaining points
            bo_lift_y_pred = [
                bo_lift.predict(row["Compound ID"]) for _, row in test_df.iterrows()
            ]
            cebo_lift_y_pred_1 = [
                cebo_lift_1.predict(row["Compound ID"]) for _, row in test_df.iterrows()
            ]
            cebo_lift_y_pred_2 = [
                cebo_lift_2.predict(row[["Compound ID", "MolLogP", "MolMR"]].to_dict())
                for _, row in test_df.iterrows()
            ]
            cebo_lift_y_pred_3 = [
                cebo_lift_3.predict(row[["Compound ID", "MolLogP", "MolMR"]].to_dict())
                for _, row in test_df.iterrows()
            ]
            cebo_lift_y_pred_4 = [
                cebo_lift_4.predict(row[["Compound ID", "Ocurrences", "SD"]].to_dict())
                for _, row in test_df.iterrows()
            ]
            cebo_lift_y_pred_5 = [
                cebo_lift_5.predict(row[["Compound ID", "Ocurrences", "SD"]].to_dict())
                for _, row in test_df.iterrows()
            ]
            # Modify results
            bo_lift_y_pred_modify = [
                sol.mean() if len(sol) >= 1 else np.nan for sol in bo_lift_y_pred
            ]
            cebo_lift_y_pred_modify_1 = [
                sol.mean() if len(sol) >= 1 else np.nan for sol in cebo_lift_y_pred_1
            ]
            cebo_lift_y_pred_modify_2 = [
                sol.mean() if len(sol) >= 1 else np.nan for sol in cebo_lift_y_pred_2
            ]
            cebo_lift_y_pred_modify_3 = [
                sol.mean() if len(sol) >= 1 else np.nan for sol in cebo_lift_y_pred_3
            ]
            cebo_lift_y_pred_modify_4 = [
                sol.mean() if len(sol) >= 1 else np.nan for sol in cebo_lift_y_pred_4
            ]
            cebo_lift_y_pred_modify_5 = [
                sol.mean() if len(sol) >= 1 else np.nan for sol in cebo_lift_y_pred_5
            ]
            # Store values
            bo_lift_result.append(
                {
                    "Iteration": i,
                    "T": T,
                    "k": k,
                    "Train": num_train,
                    "Test": num_test,
                    "Model": model,
                    "Predictions": bo_lift_y_pred_modify,
                }
            )
            cebo_lift_result_1.append(
                {
                    "Iteration": i,
                    "T": T,
                    "k": k,
                    "Train": num_train,
                    "Test": num_test,
                    "Model": model,
                    "Predictions": cebo_lift_y_pred_modify_1,
                }
            )
            cebo_lift_result_2.append(
                {
                    "Iteration": i,
                    "T": T,
                    "k": k,
                    "Train": num_train,
                    "Test": num_test,
                    "Model": model,
                    "Predictions": cebo_lift_y_pred_modify_2,
                }
            )
            cebo_lift_result_3.append(
                {
                    "Iteration": i,
                    "T": T,
                    "k": k,
                    "Train": num_train,
                    "Test": num_test,
                    "Model": model,
                    "Predictions": cebo_lift_y_pred_modify_3,
                }
            )
            cebo_lift_result_4.append(
                {
                    "Iteration": i,
                    "T": T,
                    "k": k,
                    "Train": num_train,
                    "Test": num_test,
                    "Model": model,
                    "Predictions": cebo_lift_y_pred_modify_4,
                }
            )
            cebo_lift_result_5.append(
                {
                    "Iteration": i,
                    "T": T,
                    "k": k,
                    "Train": num_train,
                    "Test": num_test,
                    "Model": model,
                    "True": list(test_df["Solubility"]),
                    "Predictions": cebo_lift_y_pred_modify_5,
                }
            )
        # Add to final results
        bo_lift_results.append(bo_lift_result)
        cebo_lift_results_1.append(cebo_lift_result_1)
        cebo_lift_results_2.append(cebo_lift_result_2)
        cebo_lift_results_3.append(cebo_lift_result_3)
        cebo_lift_results_4.append(cebo_lift_result_4)
        cebo_lift_results_5.append(cebo_lift_result_5)
    # Combine the lists into a single data structure
    ablation_data = (
        bo_lift_results,
        cebo_lift_results_1,
        cebo_lift_results_2,
        cebo_lift_results_3,
        cebo_lift_results_4,
        cebo_lift_results_5,
    )
    # Return results
    return ablation_data
