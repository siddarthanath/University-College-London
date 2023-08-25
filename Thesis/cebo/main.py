"""
This file executes the following experiments:
1. Automatic Feature Engineering (for prompt design).
2. Ablation studies.
3. Contextual Bayesian Optimisation
This is the main file which gives the results for the dissertation.
"""
# -------------------------------------------------------------------------------------------------------------------- #

# Standard Library
import itertools
import copy
import tqdm
import pickle

# Third Party
import pandas as pd
import numpy as np

# Private Party
from cebo.model import CEBO
from cebo.helper import run_experiment_cebo_lift, run_experiment_cebo_lift_other, run_experiment_cebo_lift_other_2
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


def test_run_c_bo():
    np.random.seed(42)
    import os
    os.environ["OPENAI_API_KEY"] = ""
    # Clean and store data
    bigsol_df = pd.read_csv("../cebo/data/bigsoldb.csv")
    bigsol_df = bigsol_df.dropna()
    bigsol_df = bigsol_df.drop_duplicates().reset_index(drop=True)
    bigsol_df.rename(columns={'T,K': 'Temperature'}, inplace=True)
    bigsol_df = bigsol_df.sort_values(by="SMILES")
    bigsol_df = bigsol_df.iloc[:, :-1]
    # Filter by counts
    counts = bigsol_df["SMILES"].value_counts()[::-1]
    filt = counts[(counts >= 10) & (counts <= 50)].index
    bigsol_df = bigsol_df[bigsol_df["SMILES"].isin(filt)]
    # Filter by size
    new_counts = bigsol_df["SMILES"].value_counts()[::-1]
    mask = new_counts.cumsum() <= 200
    filt = mask[mask == True].index
    bigsol_df = bigsol_df[bigsol_df["SMILES"].isin(filt)]
    bigsol_df.rename(columns={'SMILES_Solvent': 'SMILES Solvent'}, inplace=True)
    bigsol_df = bigsol_df[["SMILES", "Temperature", "SMILES Solvent", "Solubility"]]
    # Create model
    models_list = ["curie"]
    # Loop through models
    for model in models_list:
        # Parameters
        indexes = [i for i in range(bigsol_df.shape[0])]
        N = 25
        M = 1
        num_train = [2]
        # Initiate LLM
        cebo_lift = CEBO(y_name="solubility",
                         model=model,
                         selector_k=5,
                         temperature=0.7,
                         domain="chemist",
                         features=True)
        bo_lift = CEBO(y_name="solubility",
                       model=model,
                       selector_k=5,
                       temperature=0.7,
                       domain="chemist",
                       features=True)
        # Store values
        bayesOpts_bo = {}
        bayesOpts_cbo = {}
        # Acquisition functions
        aq_fns = ["expected_improvement", 'upper_confidence_bound', 'greedy']
        for i in range(len(num_train)):
            for j in range(len(aq_fns)):
                print(f"Model = {model} | Acquisition Function = {aq_fns[j]}")
                regret_bo_t_points = []
                regret_cbo_t_points = []
                for k in range(M):
                    starts = np.random.randint(0, len(indexes), M)
                    regret_bo_t, regret_cbo_t = run_experiment_cebo_lift(model_1=copy.deepcopy(bo_lift),
                                                                         model_2=copy.deepcopy(cebo_lift),
                                                                         data=bigsol_df,
                                                                         indexes=indexes,
                                                                         context="Temperature",
                                                                         target="Solubility",
                                                                         N=N,
                                                                         initial_train=num_train[i],
                                                                         aq=aq_fns[j],
                                                                         start_index=starts[k],
                                                                         )
                    regret_bo_t_points.append(regret_bo_t)
                    regret_cbo_t_points.append(regret_cbo_t)
                # Store results
                regret_bo_points = np.array(regret_bo_t_points)
                regret_cbo_points = np.array(regret_cbo_t_points)
                bayesOpts_bo[aq_fns[j]] = regret_bo_points
                bayesOpts_cbo[aq_fns[j]] = regret_cbo_points
        # Specify the file path where you want to save the pickled data
    file_path = 'bo_study.pkl'
    # Pickle and save the data
    with open(file_path, 'wb') as file:
        pickle.dump(bayesOpts_bo, file)
    print("Bayesian Optimisation completed!")
    # Specify the file path where you want to save the pickled data
    file_path = 'cbo_study.pkl'
    # Pickle and save the data
    with open(file_path, 'wb') as file:
        pickle.dump(bayesOpts_cbo, file)
    print("Contextual Bayesian Optimisation completed!")

def test_run_c_bo_2():
    import os
    os.environ["OPENAI_API_KEY"] = ""
    # Clean data
    data = pd.read_csv("../cebo/data/bigsoldb.csv")
    data = data.dropna()
    data = data.drop_duplicates().reset_index(drop=True)
    data.rename(columns={'T,K': 'Temperature'}, inplace=True)
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
        main_data = pd.concat((pd.DataFrame([sub_temp], columns=list(main_data.columns)), main_data))
    # Return smaller compound list
    sub_data = main_data.iloc[:, :6]
    mask = (sub_data.iloc[:, 1:] > 3) & (sub_data.iloc[:, 1:] < 6)
    mask = mask.all(axis=1)
    refined_data = sub_data[mask]
    refined_data = refined_data[refined_data.iloc[:, 1:].eq(5).all(axis=1)]
    refined_data = refined_data[refined_data['SMILES'].apply(lambda x: len(x) < 30)][:4]
    combined_data = data.merge(refined_data['SMILES'], on='SMILES')
    combined_data = combined_data[combined_data['Temperature'].isin(list(main_data.iloc[:, :6].columns)[1:])]
    # Final dataframe
    combined_data.rename(columns={'SMILES_Solvent': 'SMILES Solvent'}, inplace=True)
    bigsol_df = combined_data[["SMILES", "Temperature", "SMILES Solvent", "Solubility"]]
    # Create model
    models_list = ["curie"]
    # Loop through models
    for model in models_list:
        # Parameters
        indexes = [i for i in range(bigsol_df.shape[0])]
        N = 25
        M = 1
        num_train = [5]
        # Initiate LLM
        cebo_lift = CEBO(y_name="solubility",
                         model=model,
                         selector_k=5,
                         temperature=0.7,
                         domain="chemist",
                         features=True)
        bo_lift = CEBO(y_name="solubility",
                       model=model,
                       selector_k=5,
                       temperature=0.7,
                       domain="chemist",
                       features=True)
        # Store values
        bayesOpts_bo = {}
        bayesOpts_cbo = {}
        # Acquisition functions
        aq_fns = ['upper_confidence_bound', "expected_improvement", 'greedy']
        for i in range(len(num_train)):
            for j in range(len(aq_fns)):
                print(f"Model = {model} | Acquisition Function = {aq_fns[j]}")
                regret_bo_t_points = []
                regret_cbo_t_points = []
                for k in range(M):
                    starts = np.random.randint(0, len(indexes), M)
                    regret_bo_t, regret_cbo_t = run_experiment_cebo_lift(model_1=copy.deepcopy(bo_lift),
                                                                         model_2=copy.deepcopy(cebo_lift),
                                                                         data=bigsol_df,
                                                                         indexes=indexes,
                                                                         context="Temperature",
                                                                         target="Solubility",
                                                                         N=N,
                                                                         initial_train=num_train[i],
                                                                         aq=aq_fns[j],
                                                                         start_index=starts[k],
                                                                         )
                    regret_bo_t_points.append(regret_bo_t)
                    regret_cbo_t_points.append(regret_cbo_t)
                # Store results
                regret_bo_points = np.array(regret_bo_t_points)
                regret_cbo_points = np.array(regret_cbo_t_points)
                bayesOpts_bo[aq_fns[j]] = regret_bo_points
                bayesOpts_cbo[aq_fns[j]] = regret_cbo_points
        # Specify the file path where you want to save the pickled data
    file_path = 'bo_study.pkl'
    # Pickle and save the data
    with open(file_path, 'wb') as file:
        pickle.dump(bayesOpts_bo, file)
    print("Bayesian Optimisation completed!")
    # Specify the file path where you want to save the pickled data
    file_path = 'cbo_study.pkl'
    # Pickle and save the data
    with open(file_path, 'wb') as file:
        pickle.dump(bayesOpts_cbo, file)
    print("Contextual Bayesian Optimisation completed!")

"""
This file executes the following experiments:
1. Automatic Feature Engineering (for prompt design).
2. Ablation studies.
3. Contextual Bayesian Optimisation
This is the main file which gives the results for the dissertation.
"""
# -------------------------------------------------------------------------------------------------------------------- #

# Standard Library
import itertools
import copy
import tqdm
import pickle

# Third Party
import pandas as pd
import numpy as np

# Private Party
from cebo.model import CEBO
from cebo.helper import run_experiment_cebo_lift, run_experiment_cebo_lift_other
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


def test_run_c_bo():
    np.random.seed(42)
    import os
    os.environ["OPENAI_API_KEY"] = ""
    # Clean and store data
    bigsol_df = pd.read_csv("../cebo/data/bigsoldb.csv")
    bigsol_df = bigsol_df.dropna()
    bigsol_df = bigsol_df.drop_duplicates().reset_index(drop=True)
    bigsol_df.rename(columns={'T,K': 'Temperature'}, inplace=True)
    bigsol_df = bigsol_df.sort_values(by="SMILES")
    bigsol_df = bigsol_df.iloc[:, :-1]
    # Filter by counts
    counts = bigsol_df["SMILES"].value_counts()[::-1]
    filt = counts[(counts >= 10) & (counts <= 50)].index
    bigsol_df = bigsol_df[bigsol_df["SMILES"].isin(filt)]
    # Filter by size
    new_counts = bigsol_df["SMILES"].value_counts()[::-1]
    mask = new_counts.cumsum() <= 200
    filt = mask[mask == True].index
    bigsol_df = bigsol_df[bigsol_df["SMILES"].isin(filt)]
    bigsol_df.rename(columns={'SMILES_Solvent': 'SMILES Solvent'}, inplace=True)
    bigsol_df = bigsol_df[["SMILES", "Temperature", "SMILES Solvent", "Solubility"]]
    # Create model
    models_list = ["curie"]
    # Loop through models
    for model in models_list:
        # Parameters
        indexes = [i for i in range(bigsol_df.shape[0])]
        N = 25
        M = 1
        num_train = [2]
        # Initiate LLM
        cebo_lift = CEBO(y_name="solubility",
                         model=model,
                         selector_k=5,
                         temperature=0.7,
                         domain="chemist",
                         features=True)
        bo_lift = CEBO(y_name="solubility",
                       model=model,
                       selector_k=5,
                       temperature=0.7,
                       domain="chemist",
                       features=True)
        # Store values
        bayesOpts_bo = {}
        bayesOpts_cbo = {}
        # Acquisition functions
        aq_fns = ["expected_improvement", 'upper_confidence_bound', 'greedy']
        for i in range(len(num_train)):
            for j in range(len(aq_fns)):
                print(f"Model = {model} | Acquisition Function = {aq_fns[j]}")
                regret_bo_t_points = []
                regret_cbo_t_points = []
                for k in range(M):
                    starts = np.random.randint(0, len(indexes), M)
                    regret_bo_t, regret_cbo_t = run_experiment_cebo_lift(model_1=copy.deepcopy(bo_lift),
                                                                         model_2=copy.deepcopy(cebo_lift),
                                                                         data=bigsol_df,
                                                                         indexes=indexes,
                                                                         context="Temperature",
                                                                         target="Solubility",
                                                                         N=N,
                                                                         initial_train=num_train[i],
                                                                         aq=aq_fns[j],
                                                                         start_index=starts[k],
                                                                         )
                    regret_bo_t_points.append(regret_bo_t)
                    regret_cbo_t_points.append(regret_cbo_t)
                # Store results
                regret_bo_points = np.array(regret_bo_t_points)
                regret_cbo_points = np.array(regret_cbo_t_points)
                bayesOpts_bo[aq_fns[j]] = regret_bo_points
                bayesOpts_cbo[aq_fns[j]] = regret_cbo_points
        # Specify the file path where you want to save the pickled data
    file_path = 'bo_study.pkl'
    # Pickle and save the data
    with open(file_path, 'wb') as file:
        pickle.dump(bayesOpts_bo, file)
    print("Bayesian Optimisation completed!")
    # Specify the file path where you want to save the pickled data
    file_path = 'cbo_study.pkl'
    # Pickle and save the data
    with open(file_path, 'wb') as file:
        pickle.dump(bayesOpts_cbo, file)
    print("Contextual Bayesian Optimisation completed!")

def test_run_c_bo_2():
    import os
    os.environ["OPENAI_API_KEY"] = ""
    # Clean data
    data = pd.read_csv("../cebo/data/bigsoldb.csv")
    data = data.dropna()
    data = data.drop_duplicates().reset_index(drop=True)
    data.rename(columns={'T,K': 'Temperature'}, inplace=True)
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
        main_data = pd.concat((pd.DataFrame([sub_temp], columns=list(main_data.columns)), main_data))
    # Return smaller compound list
    sub_data = main_data.iloc[:, :6]
    mask = (sub_data.iloc[:, 1:] > 3) & (sub_data.iloc[:, 1:] < 6)
    mask = mask.all(axis=1)
    refined_data = sub_data[mask]
    refined_data = refined_data[refined_data.iloc[:, 1:].eq(5).all(axis=1)]
    refined_data = refined_data[refined_data['SMILES'].apply(lambda x: len(x) < 30)][:4]
    combined_data = data.merge(refined_data['SMILES'], on='SMILES')
    combined_data = combined_data[combined_data['Temperature'].isin(list(main_data.iloc[:, :6].columns)[1:])]
    # Final dataframe
    combined_data.rename(columns={'SMILES_Solvent': 'SMILES Solvent'}, inplace=True)
    bigsol_df = combined_data[["SMILES", "Temperature", "SMILES Solvent", "Solubility"]]
    # Create model
    models_list = ["curie"]
    # Loop through models
    for model in models_list:
        # Parameters
        indexes = [i for i in range(bigsol_df.shape[0])]
        N = 25
        M = 1
        num_train = [5]
        # Initiate LLM
        cebo_lift = CEBO(y_name="solubility",
                         model=model,
                         selector_k=5,
                         temperature=0.7,
                         domain="chemist",
                         features=True)
        bo_lift = CEBO(y_name="solubility",
                       model=model,
                       selector_k=5,
                       temperature=0.7,
                       domain="chemist",
                       features=True)
        # Store values
        bayesOpts_bo = {}
        bayesOpts_cbo = {}
        # Acquisition functions
        aq_fns = ['upper_confidence_bound', "expected_improvement", 'greedy']
        for i in range(len(num_train)):
            for j in range(len(aq_fns)):
                print(f"Model = {model} | Acquisition Function = {aq_fns[j]}")
                regret_bo_t_points = []
                regret_cbo_t_points = []
                for k in range(M):
                    starts = np.random.randint(0, len(indexes), M)
                    regret_bo_t, regret_cbo_t = run_experiment_cebo_lift(model_1=copy.deepcopy(bo_lift),
                                                                         model_2=copy.deepcopy(cebo_lift),
                                                                         data=bigsol_df,
                                                                         indexes=indexes,
                                                                         context="Temperature",
                                                                         target="Solubility",
                                                                         N=N,
                                                                         initial_train=num_train[i],
                                                                         aq=aq_fns[j],
                                                                         start_index=starts[k],
                                                                         )
                    regret_bo_t_points.append(regret_bo_t)
                    regret_cbo_t_points.append(regret_cbo_t)
                # Store results
                regret_bo_points = np.array(regret_bo_t_points)
                regret_cbo_points = np.array(regret_cbo_t_points)
                bayesOpts_bo[aq_fns[j]] = regret_bo_points
                bayesOpts_cbo[aq_fns[j]] = regret_cbo_points
        # Specify the file path where you want to save the pickled data
    file_path = 'bo_study.pkl'
    # Pickle and save the data
    with open(file_path, 'wb') as file:
        pickle.dump(bayesOpts_bo, file)
    print("Bayesian Optimisation completed!")
    # Specify the file path where you want to save the pickled data
    file_path = 'cbo_study.pkl'
    # Pickle and save the data
    with open(file_path, 'wb') as file:
        pickle.dump(bayesOpts_cbo, file)
    print("Contextual Bayesian Optimisation completed!")

def test_run_c_bo_3():
    import os
    os.environ["OPENAI_API_KEY"] = ""
    # Clean data
    bigsol_df = pd.read_csv("../cebo/cbo_vs_bo_df_1.csv")
    # Create model
    models_list = ["curie"]
    # Loop through models
    for model in models_list:
        # Parameters
        indexes = [i for i in range(bigsol_df.shape[0])]
        N = 25
        M = 1
        num_train = [15]
        # Initiate LLM
        cebo_lift = CEBO(y_name="solubility",
                         model=model,
                         selector_k=None,
                         temperature=0.7,
                         domain="chemist",
                         features=True)
        bo_lift = AskTellFewShotTopk(x_formatter=lambda x: f"SMILES {x[0]}, SMILES Solvent {x[1]}",
                                     y_name="solubility",
                                     y_formatter=lambda y: f"{y:.6f}",
                                     model=model,
                                     selector_k=5,
                                     temperature=0.7)
        # Store values
        bayesOpts_bo = {}
        bayesOpts_cbo = {}
        # Acquisition functions
        aq_fns = ['upper_confidence_bound']
        for i in range(len(num_train)):
            for j in range(len(aq_fns)):
                print(f"Model = {model} | Acquisition Function = {aq_fns[j]}")
                regret_bo_t_points = []
                regret_cbo_t_points = []
                for k in range(M):
                    starts = np.random.randint(0, len(indexes), M)
                    regret_bo_t, regret_cbo_t = run_experiment_cebo_lift_other(model_1=copy.deepcopy(bo_lift),
                                                                         model_2=copy.deepcopy(cebo_lift),
                                                                         data=bigsol_df,
                                                                         indexes=indexes,
                                                                         context="Temperature",
                                                                         target="Solubility",
                                                                         N=N,
                                                                         initial_train=num_train[i],
                                                                         aq=aq_fns[j],
                                                                         start_index=starts[k],
                                                                         )
                    regret_bo_t_points.append(regret_bo_t)
                    regret_cbo_t_points.append(regret_cbo_t)
                # Store results
                regret_bo_points = np.array(regret_bo_t_points)
                regret_cbo_points = np.array(regret_cbo_t_points)
                bayesOpts_bo[aq_fns[j]] = regret_bo_points
                bayesOpts_cbo[aq_fns[j]] = regret_cbo_points
        # Specify the file path where you want to save the pickled data
    file_path = 'bo_study.pkl'
    # Pickle and save the data
    with open(file_path, 'wb') as file:
        pickle.dump(bayesOpts_bo, file)
    print("Bayesian Optimisation completed!")
    # Specify the file path where you want to save the pickled data
    file_path = 'cbo_study.pkl'
    # Pickle and save the data
    with open(file_path, 'wb') as file:
        pickle.dump(bayesOpts_cbo, file)
    print("Contextual Bayesian Optimisation completed!")

def test_run_c_bo_4():
    import os
    os.environ["OPENAI_API_KEY"] = ""
    # Clean data
    bigsol_df = pd.read_csv("../cebo/cbo_vs_bo_df_1.csv")
    # Create model
    models_list = ["curie"]
    # Loop through models
    for model in models_list:
        # Parameters
        indexes = [i for i in range(bigsol_df.shape[0])]
        N = 25
        M = 1
        num_train = [15]
        # Initiate LLM
        cebo_lift = AskTellFewShotTopk(
            x_formatter=lambda x: f"SMILES {x[0]}, SMILES Solvent {x[1]} and Temperature {x[2]}",
            y_name="solubility",
            y_formatter=lambda y: f"{y:.6f}",
            model=model,
            selector_k=5,
            temperature=0.7)
        bo_lift = AskTellFewShotTopk(x_formatter=lambda x: f"SMILES {x[0]}, SMILES Solvent {x[1]}",
                                     y_name="solubility",
                                     y_formatter=lambda y: f"{y:.6f}",
                                     model=model,
                                     selector_k=5,
                                     temperature=0.7)
        # Store values
        bayesOpts_bo = {}
        bayesOpts_cbo = {}
        # Acquisition functions
        aq_fns = ['upper_confidence_bound']
        for i in range(len(num_train)):
            for j in range(len(aq_fns)):
                print(f"Model = {model} | Acquisition Function = {aq_fns[j]}")
                regret_bo_t_points = []
                regret_cbo_t_points = []
                for k in range(M):
                    starts = np.random.randint(0, len(indexes), M)
                    regret_bo_t, regret_cbo_t = run_experiment_cebo_lift_other_2(model_1=copy.deepcopy(bo_lift),
                                                                         model_2=copy.deepcopy(cebo_lift),
                                                                         data=bigsol_df,
                                                                         indexes=indexes,
                                                                         context="Temperature",
                                                                         target="Solubility",
                                                                         N=N,
                                                                         initial_train=num_train[i],
                                                                         aq=aq_fns[j],
                                                                         start_index=starts[k],
                                                                         )
                    regret_bo_t_points.append(regret_bo_t)
                    regret_cbo_t_points.append(regret_cbo_t)
                # Store results
                regret_bo_points = np.array(regret_bo_t_points)
                regret_cbo_points = np.array(regret_cbo_t_points)
                bayesOpts_bo[aq_fns[j]] = regret_bo_points
                bayesOpts_cbo[aq_fns[j]] = regret_cbo_points
        # Specify the file path where you want to save the pickled data
    file_path = 'bo_study.pkl'
    # Pickle and save the data
    with open(file_path, 'wb') as file:
        pickle.dump(bayesOpts_bo, file)
    print("Bayesian Optimisation completed!")
    # Specify the file path where you want to save the pickled data
    file_path = 'cbo_study.pkl'
    # Pickle and save the data
    with open(file_path, 'wb') as file:
        pickle.dump(bayesOpts_cbo, file)
    print("Contextual Bayesian Optimisation completed!")
