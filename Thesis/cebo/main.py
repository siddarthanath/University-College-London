"""
This file executes the following experiments:
1. Automatic Feature Engineering (for prompt design).
2. Ablation studies.
3. Contextual Bayesian Optimisation
This is the main file which gives the results for the dissertation.
"""
# -------------------------------------------------------------------------------------------------------------------- #

# Standard Library
import os
import itertools
import pickle

# Third Party
import pandas as pd
import numpy as np

# Private Party
from model import CEBO
from helper import run_experiment_cebo_lift_main
from bolift import AskTellFewShotTopk


# -------------------------------------------------------------------------------------------------------------------- #
def test_run_tell_predict():
    import os
    os.environ["OPENAI_API_KEY"] = ""
    # Clean and store data
    aqsoldb_df = pd.read_csv("../cebo/data/aqsoldb.csv")
    aqsoldb_df = aqsoldb_df.dropna()
    aqsoldb_df = aqsoldb_df.drop_duplicates().reset_index(drop=True)
    aqsoldb_df.rename(columns={'Name': 'Compound ID'}, inplace=True)
    aqsoldb_df = aqsoldb_df.drop(["ID"], axis=1)
    aqsoldb_df = aqsoldb_df[aqsoldb_df["Compound ID"].str.len() < 15].reset_index(drop=True)
    aqsoldb_df = aqsoldb_df.sample(n=1000, random_state=42)
    # Hyperparameters
    T_list = [0.7]
    k_list = [5]
    train_num_list = [5, 15, 25, 35, 45]
    test_num_list = [10]
    models_list = ["curie", "davinci"]
    # Store results
    bo_lift_results = []
    cebo_lift_results_1 = []
    cebo_lift_results_2 = []
    cebo_lift_results_3 = []
    cebo_lift_results_4 = []
    # Loop
    for T, k, num_train, num_test, model in itertools.product(T_list, k_list, train_num_list, test_num_list,
                                                              models_list):
        bo_lift_result = []
        cebo_lift_result_1 = []
        cebo_lift_result_2 = []
        cebo_lift_result_3 = []
        cebo_lift_result_4 = []
        for i in range(10):
            # Create data
            shuffled_df = aqsoldb_df.sample(frac=1, random_state=42)
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
            # Tell some points to the model
            for _, row in train_df.iterrows():
                bo_lift.tell(row["Compound ID"], row["Solubility"])
                cebo_lift_1.tell(row["Compound ID"], row["Solubility"])
                cebo_lift_2.tell(row[["Compound ID", "MolLogP", "MolMR", "Solubility"]].to_dict())
                cebo_lift_3.tell(row[["Compound ID", "MolLogP", "MolMR", "Solubility"]].to_dict())
                cebo_lift_4.tell(row[["Compound ID", "Ocurrences", "SD", "Solubility"]].to_dict())
            # Predict remaining points
            bo_lift_y_pred = [bo_lift.predict(row["Compound ID"]) for _, row in test_df.iterrows()]
            cebo_lift_y_pred_1 = [cebo_lift_1.predict(row["Compound ID"]) for _, row in test_df.iterrows()]
            cebo_lift_y_pred_2 = [cebo_lift_2.predict(row[["Compound ID", "MolLogP", "MolMR"]].to_dict()) for _, row
                                  in test_df.iterrows()]
            cebo_lift_y_pred_3 = [cebo_lift_3.predict(row[["Compound ID", "MolLogP", "MolMR"]].to_dict()) for _, row
                                  in test_df.iterrows()]
            cebo_lift_y_pred_4 = [cebo_lift_4.predict(row[["Compound ID", "Ocurrences", "SD"]].to_dict()) for _, row
                                  in test_df.iterrows()]
            # Modify results
            bo_lift_y_pred_modify = [sol.mean() if len(sol) >= 1 else np.nan for sol in bo_lift_y_pred]
            cebo_lift_y_pred_modify_1 = [sol.mean() if len(sol) >= 1 else np.nan for sol in cebo_lift_y_pred_1]
            cebo_lift_y_pred_modify_2 = [sol.mean() if len(sol) >= 1 else np.nan for sol in cebo_lift_y_pred_2]
            cebo_lift_y_pred_modify_3 = [sol.mean() if len(sol) >= 1 else np.nan for sol in cebo_lift_y_pred_3]
            cebo_lift_y_pred_modify_4 = [sol.mean() if len(sol) >= 1 else np.nan for sol in cebo_lift_y_pred_4]
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
        # Add to final results
        bo_lift_results.append(bo_lift_result)
        cebo_lift_results_1.append(cebo_lift_result_1)
        cebo_lift_results_2.append(cebo_lift_result_2)
        cebo_lift_results_3.append(cebo_lift_result_3)
        cebo_lift_results_4.append(cebo_lift_result_4)
    # Combine the lists into a single data structure
    pickle_data = (bo_lift_results, cebo_lift_results_1,
                   cebo_lift_results_2, cebo_lift_results_3,
                   cebo_lift_results_4)

    # Specify the file path where you want to save the pickled data
    file_path = 'ablation_study.pkl'
    # Pickle and save the data
    with open(file_path, 'wb') as file:
        pickle.dump(pickle_data, file)


def test_run_bo_vs_c_bo():
    os.environ["OPENAI_API_KEY"] = ""
    # Clean data
    bigsoldb_df = pd.read_csv("/Users/siddarthanath/Documents/University-College-London/Thesis/cebo/data/cbo_vs_bo_df_1"
                              ".csv")
    # Create model
    models_list = ["curie"]
    bayesOpts_results = None
    # Loop through models
    for model in models_list:
        # Parameters
        indexes = [i for i in range(bigsoldb_df.shape[0])]
        N = 25
        M = 3
        num_train = [15]
        # BO-LIFT with MMR (BO)
        bo_lift_1 = AskTellFewShotTopk(x_formatter=lambda x: f"SMILES {x[0]}, SMILES Solvent {x[1]}",
                                       y_name="solubility",
                                       y_formatter=lambda y: f"{y:.5f}",
                                       model=model,
                                       selector_k=1,
                                       temperature=0.7)
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
            selector_k=1,
            temperature=0.7)
        # BO-LIFT without MMR (C-BO)
        bo_lift_4 = AskTellFewShotTopk(
            x_formatter=lambda x: f"SMILES {x[0]}, SMILES Solvent {x[1]} and Temperature {x[2]}",
            y_name="solubility",
            y_formatter=lambda y: f"{y:.5f}",
            model=model,
            selector_k=5,
            temperature=0.7)
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
                    framework_types = {"BO": {"BO-LIFT": [bo_lift_1, bo_lift_2]},
                                       "C-BO": {"BO-LIFT": [bo_lift_3, bo_lift_4]}}
                    starts = np.random.randint(0, len(indexes), M)
                    regret_results = run_experiment_cebo_lift_main(frameworks=framework_types,
                                                                   data=bigsoldb_df,
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
    # Specify the file path where you want to save the pickled data
    file_path = 'bo_vs_cbo_study.pkl'
    # Pickle and save the data
    with open(file_path, 'wb') as file:
        pickle.dump(bayesOpts_results, file)
    print("Contextual Bayesian Optimisation completed!")
