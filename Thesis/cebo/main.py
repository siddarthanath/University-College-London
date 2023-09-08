"""
This file executes the following experiments:
1. Automatic Feature Engineering (for prompt design).
2. Ablation studies.
3. Contextual Bayesian Optimisation.
This is the main file which gives the results for the dissertation.
"""
# -------------------------------------------------------------------------------------------------------------------- #

# Standard Library
import os
import itertools
import pickle

# Third Party
import numpy as np

# Private Party
from .model import CEBO
from .helper import run_experiment_cebo_lift_main
from .bo_lift import AskTellFewShotTopk


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
    for T, k, num_train, num_test, model in itertools.product(T_list, k_list, train_num_list, test_num_list,
                                                              models_list):
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
            bo_lift = AskTellFewShotTopk(x_formatter=lambda x: f"compound id {x}",
                                         y_name="solubility",
                                         y_formatter=lambda y: f"{y:.6f}",
                                         model=model,
                                         selector_k=k,
                                         temperature=0.7)
            cebo_lift_1 = AskTellFewShotTopk(x_formatter=lambda x: f"compound id {x}",
                                             y_name="solubility",
                                             y_formatter=lambda y: f"{y:.6f}",
                                             model=model,
                                             selector_k=k,
                                             temperature=T,
                                             prefix=(f"You are an expert chemist. "
                                                     "The following are correctly answered questions. "
                                                     "Each answer is numeric and ends with ###\n"))
            cebo_lift_2 = CEBO(y_name="solubility",
                               model=model,
                               selector_k=k,
                               temperature=T,
                               domain=None,
                               features=True)
            cebo_lift_3 = CEBO(y_name="solubility",
                               model=model,
                               selector_k=k,
                               temperature=T,
                               domain="chemist",
                               features=True)
            cebo_lift_4 = CEBO(y_name="solubility",
                               model=model,
                               selector_k=k,
                               temperature=T,
                               domain="chemist",
                               features=True)
            cebo_lift_5 = CEBO(y_name="solubility",
                               model=model,
                               selector_k=k,
                               temperature=T,
                               domain="chemist",
                               features=True)
            # Tell some points to the model
            for _, row in train_df.iterrows():
                bo_lift.tell(row["Compound ID"], row["Solubility"])
                cebo_lift_1.tell(row["Compound ID"], row["Solubility"])
                cebo_lift_2.tell(row[["Compound ID", "MolLogP", "MolMR", "Solubility"]].to_dict())
                cebo_lift_3.tell(row[["Compound ID", "MolLogP", "MolMR", "Solubility"]].to_dict())
                cebo_lift_4.tell(row[["Compound ID", "Ocurrences", "SD", "Solubility"]].to_dict())
                cebo_lift_5.tell(row[["Compound ID", "Ocurrences", "SD", "Solubility"]].to_dict())
            # Predict remaining points
            bo_lift_y_pred = [bo_lift.predict(row["Compound ID"]) for _, row in test_df.iterrows()]
            cebo_lift_y_pred_1 = [cebo_lift_1.predict(row["Compound ID"]) for _, row in test_df.iterrows()]
            cebo_lift_y_pred_2 = [cebo_lift_2.predict(row[["Compound ID", "MolLogP", "MolMR"]].to_dict()) for _, row
                                  in test_df.iterrows()]
            cebo_lift_y_pred_3 = [cebo_lift_3.predict(row[["Compound ID", "MolLogP", "MolMR"]].to_dict()) for _, row
                                  in test_df.iterrows()]
            cebo_lift_y_pred_4 = [cebo_lift_4.predict(row[["Compound ID", "Ocurrences", "SD"]].to_dict()) for _, row
                                  in test_df.iterrows()]
            cebo_lift_y_pred_5 = [cebo_lift_5.predict(row[["Compound ID", "Ocurrences", "SD"]].to_dict()) for _, row
                                  in test_df.iterrows()]
            # Modify results
            bo_lift_y_pred_modify = [sol.mean() if len(sol) >= 1 else np.nan for sol in bo_lift_y_pred]
            cebo_lift_y_pred_modify_1 = [sol.mean() if len(sol) >= 1 else np.nan for sol in cebo_lift_y_pred_1]
            cebo_lift_y_pred_modify_2 = [sol.mean() if len(sol) >= 1 else np.nan for sol in cebo_lift_y_pred_2]
            cebo_lift_y_pred_modify_3 = [sol.mean() if len(sol) >= 1 else np.nan for sol in cebo_lift_y_pred_3]
            cebo_lift_y_pred_modify_4 = [sol.mean() if len(sol) >= 1 else np.nan for sol in cebo_lift_y_pred_4]
            cebo_lift_y_pred_modify_5 = [sol.mean() if len(sol) >= 1 else np.nan for sol in cebo_lift_y_pred_5]
            # Store values
            bo_lift_result.append({"Iteration": i,
                                   "T": T,
                                   "k": k,
                                   "Train": num_train,
                                   "Test": num_test,
                                   "Model": model,
                                   "Predictions": bo_lift_y_pred_modify
                                   })
            cebo_lift_result_1.append({"Iteration": i,
                                       "T": T,
                                       "k": k,
                                       "Train": num_train,
                                       "Test": num_test,
                                       "Model": model,
                                       "Predictions": cebo_lift_y_pred_modify_1
                                       })
            cebo_lift_result_2.append({"Iteration": i,
                                       "T": T,
                                       "k": k,
                                       "Train": num_train,
                                       "Test": num_test,
                                       "Model": model,
                                       "Predictions": cebo_lift_y_pred_modify_2
                                       })
            cebo_lift_result_3.append({"Iteration": i,
                                       "T": T,
                                       "k": k,
                                       "Train": num_train,
                                       "Test": num_test,
                                       "Model": model,
                                       "Predictions": cebo_lift_y_pred_modify_3
                                       })
            cebo_lift_result_4.append({"Iteration": i,
                                       "T": T,
                                       "k": k,
                                       "Train": num_train,
                                       "Test": num_test,
                                       "Model": model,
                                       "Predictions": cebo_lift_y_pred_modify_4
                                       })
            cebo_lift_result_5.append({"Iteration": i,
                                       "T": T,
                                       "k": k,
                                       "Train": num_train,
                                       "Test": num_test,
                                       "Model": model,
                                       "True": list(test_df["Solubility"]),
                                       "Predictions": cebo_lift_y_pred_modify_5
                                       })
        # Add to final results
        bo_lift_results.append(bo_lift_result)
        cebo_lift_results_1.append(cebo_lift_result_1)
        cebo_lift_results_2.append(cebo_lift_result_2)
        cebo_lift_results_3.append(cebo_lift_result_3)
        cebo_lift_results_4.append(cebo_lift_result_4)
        cebo_lift_results_5.append(cebo_lift_result_5)
    # Combine the lists into a single data structure
    ablation_data = (bo_lift_results, cebo_lift_results_1,
                     cebo_lift_results_2, cebo_lift_results_3,
                     cebo_lift_results_4, cebo_lift_results_5)
    # Return results
    return ablation_data


def run_bo_vs_c_bo(data, N, M, num_train, models_list):
    # Create model
    bayesOpts_results = None
    # Loop through models
    for model in models_list:
        # Parameters
        indexes = [i for i in range(data.shape[0])]
        # Store values
        bayesOpts_results = {}
        # Acquisition functions
        aq_fns = ['upper_confidence_bound']
        for i in range(len(num_train)):
            for j in range(len(aq_fns)):
                print(f"Model = {model} | Acquisition Function = {aq_fns[j]}")
                regret_total_results = []
                for k in range(M):
                    # Model lists
                    # BO-LIFT with MMR (BO)
                    bo_lift_1 = AskTellFewShotTopk(x_formatter=lambda x: f"SMILES {x[0]}, SMILES Solvent {x[1]}",
                                                   y_name="solubility",
                                                   y_formatter=lambda y: f"{y:.5f}",
                                                   model=model,
                                                   selector_k=5,
                                                   temperature=0.5)
                    # BO-LIFT without MMR (BO)
                    bo_lift_2 = AskTellFewShotTopk(x_formatter=lambda x: f"SMILES {x[0]}, SMILES Solvent {x[1]}",
                                                   y_name="solubility",
                                                   y_formatter=lambda y: f"{y:.5f}",
                                                   model=model,
                                                   selector_k=5,
                                                   temperature=0.7)
                    # CEBO-LIFT with MMR (BO)
                    cebo_lift_1 = CEBO(y_name="solubility",
                                       model=model,
                                       selector_k=1,
                                       temperature=0.7,
                                       domain="chemist",
                                       features=True)
                    # CEBO-LIFT without MMR (BO)
                    cebo_lift_2 = CEBO(y_name="solubility",
                                       model=model,
                                       selector_k=5,
                                       temperature=0.7,
                                       domain="chemist",
                                       features=True)
                    # BO-LIFT with MMR (C-BO)
                    bo_lift_3 = AskTellFewShotTopk(
                        x_formatter=lambda x: f"SMILES {x[0]}, SMILES Solvent {x[1]} and Temperature {x[2]}",
                        y_name="solubility",
                        y_formatter=lambda y: f"{y:.5f}",
                        model=model,
                        selector_k=5,
                        temperature=0.5)
                    # BO-LIFT without MMR (C-BO)
                    bo_lift_4 = AskTellFewShotTopk(
                        x_formatter=lambda x: f"SMILES {x[0]}, SMILES Solvent {x[1]} and Temperature {x[2]}",
                        y_name="solubility",
                        y_formatter=lambda y: f"{y:.5f}",
                        model=model,
                        selector_k=5,
                        temperature=0.75)
                    # CEBO-LIFT with MMR (C-BO)
                    cebo_lift_3 = CEBO(y_name="solubility",
                                       model=model,
                                       selector_k=1,
                                       temperature=0.7,
                                       domain="chemist",
                                       features=True)
                    # CEBO-LIFT without MMR (C-BO)
                    cebo_lift_4 = CEBO(y_name="solubility",
                                       model=model,
                                       selector_k=5,
                                       temperature=0.7,
                                       domain="chemist",
                                       features=True)
                    framework_types = {"BO": {"BO-LIFT": [bo_lift_1]},
                                       "C-BO": {"BO-LIFT": [bo_lift_3]}}
                    starts = np.random.randint(0, len(indexes), M)
                    regret_results = run_experiment_cebo_lift_main(frameworks=framework_types,
                                                                   data=data,
                                                                   indexes=indexes,
                                                                   context="Temperature",
                                                                   target="Solubility",
                                                                   N=N,
                                                                   initial_train=num_train[i],
                                                                   aq=aq_fns[j],
                                                                   start_index=starts[k],
                                                                   )
                    regret_total_results.append(regret_results)
                # Store results
                bayesOpts_results[aq_fns[j]] = regret_total_results
    # Return results
    return bayesOpts_results
