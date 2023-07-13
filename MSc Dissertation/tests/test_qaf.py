"""
This file tests the QAF interface Vs BO-Lift interface.
"""
# -------------------------------------------------------------------------------------------------------------------- #

# Standard Library
import os

# Third Party
import numpy as np
import pandas as pd
import pickle

# Private
import qafnet
import bolift
import qafnet

import uncertainty_toolbox as uct
import cloudpickle
from bolift.asktell import PromptTemplate
import copy

os.environ["OPENAI_API_KEY"] = "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"
def combine(s, l):
    '''Number of combinations of l elements with max = s'''
    return (s ** l - (s - 1) ** (l))


def prob(s, l, n):
    '''Probability of getting a sample with max([x0,x1,...,xl]) = s where xi={0,n}'''
    return combine(s, l) * ((1 / n) ** l)


def expected_value_p(l, n):
    '''Expected value of max([x0,x1,...,xl]) where xi={0,n}'''
    E = [s * prob(s, l, n) for s in range(1, 100 + 1)]
    return sum(E)


def expected_value_q(l, n, data):
    '''Expected value of max([x0,x1,...,xl]) where xi={0,n}'''
    quants = [data.quantile(i / 100) for i in range(100 + 1)]
    # E = [(quants[s-1]) * prob(s, l, n) for s in range(1,100+1)]
    E = [((quants[s - 1] + quants[s]) / 2) * prob(s, l, n) for s in range(1, 100 + 1)]
    return sum(E)


from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def run_experiment(
        asktell, pool, raw_data, indexes, x_name, y_name, N=15, initial_train=1, ask_K=1, aq="random",
        start_index=0, calibrate=False
):
    data = raw_data.iloc[:initial_train]
    if aq == 'random_mean':
        return [(i, expected_value_q(i, 100, raw_data[y_name])) for i in range(1, N + 1)]
    asktell.tell(data=data)
    if calibrate:
        y = data['Solubility'].tolist()
        pred = [asktell.predict(row[:-1]) for _, row in data.iterrows()]
        ymeans = np.array([yhi[0].mean() for yhi in pred])
        ystds = np.array([yhi[0].std() for yhi in pred])
        calibration_factor = uct.recalibration.optimize_recalibration_ratio(ymeans, ystds, np.array(y),
                                                                            criterion="miscal")
        asktell.set_calibration_factor(calibration_factor)

    x = copy.deepcopy(raw_data).iloc[:, :-1]
    pool.reset()
    xi = x.iloc[start_index].to_dict()
    x.drop(start_index, inplace=True)
    pool.choose(xi)
    # give one point
    asktell.tell(data=raw_data[raw_data[x_name] == xi[x_name]])
    point = [(xi[x_name], raw_data.iloc[start_index]['Solubility'])]
    best = point[0][1]
    for i in range(1, N):
        if i == N - 1 and aq != "random":
            aq = "greedy"
        px, _, py = asktell.ask(pool, k=ask_K, aq_fxn=aq, _lambda=1.0)
        for j in range(ask_K):
            xc = px[j]
            x.drop(x[x[x_name]==xc[x_name]].index[0], inplace=True)
            pool.choose(xc)
            y = float(raw_data[raw_data[x_name] == xc[x_name]].iloc[0][-1])
            asktell.tell(data=raw_data[raw_data[x_name] == xc[x_name]])
            best = max(y, best)
        point.append((xc, best))
    return point

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def run_experiment_2(
    asktell, pool, raw_data, indexes, x_name, y_name, N=10, initial_train=1, ask_K=1, aq="random", start_index=0, calibrate=False
):
    if aq=='random_mean':
       return [ (i, expected_value_q(i, 100, raw_data[y_name])) for i in range(1,N+1) ]
    for i in indexes[:initial_train]:
        asktell.tell(raw_data[x_name].iloc[i], float(raw_data[y_name].iloc[i]))
    if calibrate:
        y = [float(raw_data[y_name].iloc[i]) for i in indexes[:initial_train]]
        pred = asktell.predict(y)
        ymeans = np.array([yhi.mean() for yhi in pred])
        ystds = np.array([yhi.std() for yhi in pred])
        calibration_factor = uct.recalibration.optimize_recalibration_ratio(ymeans, ystds, np.array(y),
                                                                        criterion="miscal")
        asktell.set_calibration_factor(calibration_factor)

    x = [raw_data[x_name].iloc[i] for i in indexes]

    pool.reset()
    xi = x[start_index]
    x.remove(xi)
    pool.choose(xi)
    # give one point
    yi = float(raw_data[raw_data[x_name] == xi][y_name].iloc[0])
    asktell.tell(xi, yi)
    point = [(xi, yi)]
    best = point[0][1]
    for i in range(1, N):
        if i == N - 1 and aq != "random":
            aq = "greedy"
        px, _, py = asktell.ask(pool, k=ask_K, aq_fxn=aq, _lambda=1.0)
        for j in range(ask_K):
          xc = px[j]
          x.remove(xc)
          pool.choose(xc)
          y = float(raw_data[raw_data[x_name] == xc][y_name].iloc[0])
          asktell.tell(xc, y)
          best = max(y, best)
        point.append((xc, best))
    return point
def test_bo():
    np.random.seed(0)

    data_path = "../paper/full_df.csv"
    # aqsol_df = pd.read_csv(data_path)
    # # Clean
    # aqsol_df = aqsol_df.dropna()
    # aqsol_df = aqsol_df.drop_duplicates().reset_index(drop=True)
    # aqsol_df.rename(columns={'Name': 'Compound ID'}, inplace=True)
    # aqsol_df = aqsol_df.drop(["ID"], axis=1)
    # aqsol_df = aqsol_df[aqsol_df["Compound ID"].str.len() < 15].reset_index(drop=True)
    # cols_to_keep = []
    # for i in range(len(aqsol_df.columns)):
    #     if aqsol_df.dtypes[aqsol_df.columns[i]] == 'int64' or aqsol_df.dtypes[aqsol_df.columns[i]] == 'float64':
    #         cols_to_keep.append(aqsol_df.columns[i])
    # compound_df = aqsol_df["Compound ID"]
    # features_df = aqsol_df[cols_to_keep]
    # # Create full dataset for BO-Lift
    # full_df = pd.concat((compound_df, features_df), axis=1)
    # raw_data = full_df.sample(n=50)
    # raw_data = raw_data.reset_index(drop=True)
    raw_data = pd.read_csv(data_path)
    raw_data = raw_data.iloc[:, 1:]
    columns = ['Compound ID', 'MolLogP', 'MolMR', 'BertzCT', 'Solubility']
    raw_data = raw_data[columns]

    # raw_data['measured log(solubility:mol/L)'] = -raw_data['measured log(solubility:mol/L)']

    N = raw_data.shape[0]
    indexes = [i for i in range(N)]  # np.random.choice(raw_data.shape[0], int(N), replace=False)
    x_name = "Compound ID"
    y_name = "solubility"

    # asktell = bolift.AskTellFewShotTopk(
    #     prefix="",
    #     prompt_template=PromptTemplate(
    #         input_variables=["x", "y", "y_name"],
    #         template="Q: What is the {y_name} of {x}?@@@\nA: {y}###",
    #     ),
    #     suffix="What is the {y_name} of {x}?@@@\nA:",
    #     x_formatter=lambda x: f"iupac name {x}",
    #     y_name="measured log solubility in mols per litre",
    #     y_formatter=lambda y: f"{y:.2f}",
    #     model="text-davinci-003",
    #     selector_k=5,
    #     temperature=0.7,)

    asktell = qafnet.QAFFewShotTopK(y_name=y_name,
                                    model="text-davinci-003",
                                    selector_k=5,
                                    temperature=0.7)
    path_random = "../paper/out/sol - random.pkl"
    path = "../paper/out/sol_davinci_100.pkl"
    pool_path = "../paper/out/sol_pool_msc_1.pkl"
    # Removed pool path
    pool = qafnet.Pool(raw_data.iloc[:, :-1].to_dict(orient='records'))
    cloudpickle.dump(pool, open(pool_path, "wb"))
    N = 15
    M = 5
    starts = np.random.randint(0, len(indexes), M)

    bayesOpts = {}

    for aq in ["random", "expected_improvement", 'upper_confidence_bound']:
        print(aq, "start:", end=" ")
        points = []
        for i in range(M):
            print(i, end=",  ")
            point = run_experiment(
                copy.deepcopy(asktell),
                copy.deepcopy(pool),
                raw_data,
                indexes,
                x_name,
                y_name,
                N=N,
                aq=aq,
                start_index=starts[i],
                calibrate=True,
                initial_train=25
            )
            points.append(point)
        # plot mean
        points = np.array(points)
        bayesOpts[aq] = points
        print(aq, "done")
    # Specify the file path where you want to save the pickled data
    file_path = 'results/bo-protocol-test-1/qafnet-bo-davinci'
    # Open the file in binary mode
    with open(file_path, 'wb') as file:
        # Pickle the data and write it to the file
        pickle.dump(bayesOpts, file)

def test_bo_2():
    np.random.seed(0)

    data_path = "../paper/data/full_solubility.csv"
    aqsol_df = pd.read_csv(data_path)
    # Clean
    aqsol_df = aqsol_df.dropna()
    aqsol_df = aqsol_df.drop_duplicates().reset_index(drop=True)
    aqsol_df.rename(columns={'Name': 'Compound ID'}, inplace=True)
    aqsol_df = aqsol_df.drop(["ID"], axis=1)
    aqsol_df = aqsol_df[aqsol_df["Compound ID"].str.len() < 15].reset_index(drop=True)
    cols_to_keep = []
    for i in range(len(aqsol_df.columns)):
        if aqsol_df.dtypes[aqsol_df.columns[i]] == 'int64' or aqsol_df.dtypes[aqsol_df.columns[i]] == 'float64':
            cols_to_keep.append(aqsol_df.columns[i])
    compound_df = aqsol_df["Compound ID"]
    features_df = aqsol_df[cols_to_keep]
    # Create full dataset for BO-Lift
    full_df = pd.concat((compound_df, features_df), axis=1)
    full_df = full_df.sample(n=200).reset_index(drop=True)
    raw_data = copy.deepcopy(full_df)
    raw_data = raw_data.rename(columns={'Solubility': 'solubility'})
    columns = ['Compound ID', 'solubility']
    raw_data = raw_data[columns]
    N = raw_data.shape[0]
    indexes = [i for i in range(N)]  # np.random.choice(raw_data.shape[0], int(N), replace=False)
    x_name = "Compound ID"
    y_name = "solubility"
    print(len(raw_data), len(indexes))

    asktell = bolift.AskTellFewShotTopk(
        prefix="",
        prompt_template=PromptTemplate(
            input_variables=["x", "y", "y_name"],
            template="Q: What is the {y_name} of {x}?@@@\nA: {y}###",
        ),
        suffix="What is the {y_name} of {x}?@@@\nA:",
        x_formatter=lambda x: f"iupac name {x}",
        y_name="measured log solubility in mols per litre",
        y_formatter=lambda y: f"{y:.2f}",
        model="text-davinci-003",
        selector_k=5,
        temperature=0.7,
    )
    x = [raw_data[x_name].iloc[i] for i in indexes]
    pool = bolift.Pool(list(x), formatter=lambda x: f"experimental procedure: {x}")
    N = 15
    M = 5
    starts = np.random.randint(0, len(indexes), M)
    bayesOpts = {}
    for aq in ["expected_improvement", 'upper_confidence_bound']:
        print(aq, "start:", end=" ")
        points = []
        for i in range(M):
            print(i, end=",  ")
            point = run_experiment_2(
                copy.deepcopy(asktell),
                copy.deepcopy(pool),
                raw_data,
                indexes,
                x_name,
                y_name,
                N=N,
                aq=aq,
                start_index=starts[i],
                calibrate=True,
                initial_train=25
            )
            points.append(point)
        # plot mean
        points = np.array(points)
        bayesOpts[aq] = points
        print(aq, "done")
    # Specify the file path where you want to save the pickled data
    file_path = 'results/bo-protocol-test-1/bolift-bo-davinci'
    # Open the file in binary mode
    with open(file_path, 'wb') as file:
        # Pickle the data and write it to the file
        pickle.dump(bayesOpts, file)






@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def run_experiment_qafnet_bo(
        asktell, pool, raw_data, x_name, y_name, N=15, initial_train=1, ask_K=1, aq="random",
        start_index=0, calibrate=False
):
    data = raw_data.iloc[:initial_train]
    if aq == 'random_mean':
        return [(i, expected_value_q(i, 100, raw_data[y_name])) for i in range(1, N + 1)]
    asktell.tell(data=data)
    if calibrate and type(asktell._calibration_factor) is type(None):
        y = data['Solubility'].tolist()
        pred = [asktell.predict(row[:-1]) for _, row in data.iterrows()]
        ymeans = np.array([yhi[0].mean() for yhi in pred])
        ystds = np.array([yhi[0].std() for yhi in pred])
        calibration_factor = uct.recalibration.optimize_recalibration_ratio(ymeans, ystds, np.array(y),
                                                                            criterion="miscal")
        asktell.set_calibration_factor(calibration_factor)
    # Initiate Pool
    x = copy.deepcopy(raw_data).iloc[:, :-1]
    pool.reset()
    xi = x.iloc[start_index].to_dict()
    x.drop(start_index, inplace=True)
    pool.choose(xi)
    asktell.tell(data=raw_data[raw_data[x_name] == xi[x_name]])
    point = [(xi[x_name], raw_data.iloc[start_index]['Solubility'])]
    best = point[0][1]
    # Initiate Bayesian Optimisation Protocol
    for i in range(1, N):
        if i == N - 1 and aq != "random":
            aq = "greedy"
        px, _, py = asktell.ask(pool, k=ask_K, aq_fxn=aq, _lambda=1.0)
        for j in range(ask_K):
            xc = px[j]
            x.drop(x[x[x_name]==xc[x_name]].index[0], inplace=True)
            pool.choose(xc)
            y = float(raw_data[raw_data[x_name] == xc[x_name]].iloc[0][-1])
            asktell.tell(data=raw_data[raw_data[x_name] == xc[x_name]])
            best = max(y, best)
        point.append((xc, best))
    return point




def test_qafnet_bo_1():
    num_samples = 200
    initial_train = 25
    data_path = "/Users/siddarthanath/Documents/University-College-London/MSc Dissertation/paper/data/bo_protocol_data.csv"
    model = "text-davinci-003"
    assert initial_train <= num_samples // 2
    # Data process
    raw_data = pd.read_csv(data_path)
    raw_data = raw_data.iloc[:, 1:]
    columns = ['Compound ID', 'MolLogP', 'MolMR', 'BertzCT', 'Solubility']
    raw_data = raw_data[columns]
    # Parameters
    indexes = [i for i in range(raw_data.shape[0])]
    x_name = "Compound ID"
    y_name = "solubility"
    # Initiate LLM
    asktell = qafnet.QAFFewShotTopK(y_name=y_name,
                                    model=model,
                                    selector_k=5,
                                    temperature=0.7)
    pool = 1
    N = 15
    M = 5
    starts = np.random.randint(0, len(indexes), M)
    bayesOpts = {}
    aq_fns = ["expected_improvement", 'upper_confidence_bound', 'random_mean']
    for i in range(len(aq_fns)):
        points = []
        for j in range(M):
            point = run_experiment_qafnet_bo(
                copy.deepcopy(asktell),
                copy.deepcopy(pool),
                raw_data,
                indexes,
                x_name,
                y_name,
                N=N,
                aq=aq_fns[i],
                start_index=starts[j],
                calibrate=True,
                initial_train=initial_train
            )
            points.append(point)
        # Store results
        points = np.array(points)
        bayesOpts[aq_fns[i]] = points
    # Specify the base directory
    base_directory = './results'
    # Specify the base file path
    base_file_path = 'bo-protocol-test'
    file_name = f'qafnet-bo-{model}'
    # Initialize the index
    index = 1
    # Check if the file path exists
    while os.path.exists(f"{base_directory}/{base_file_path}-{index}/{file_name}"):
        index += 1
    # Create the new directory path
    new_directory_path = f"{base_directory}/{base_file_path}-{index}"
    os.makedirs(new_directory_path, exist_ok=True)
    # Create the new file path
    new_file_path = f"{new_directory_path}/{file_name}"
    # Create the new file and pickle the data
    with open(new_file_path, 'wb') as file:
        pickle.dump(bayesOpts, file)


def test_paper():
    np.random.seed(0)
    # Establish path to solubility data
    esol_df = pd.read_csv("../paper/data/esol_iupac.csv")
    aqsol_df = pd.read_csv("../paper/data/full_solubility.csv")
    # Clean
    aqsol_df = aqsol_df.dropna()
    aqsol_df = aqsol_df.drop_duplicates().reset_index(drop=True)
    aqsol_df.rename(columns={'Name': 'Compound ID'}, inplace=True)
    esol_df = esol_df.dropna()
    esol_df = esol_df.drop_duplicates().reset_index(drop=True)
    final_df = aqsol_df.merge(esol_df, on='Compound ID')
    final_df = final_df.drop(["SMILES_x"], axis=1)
    final_df.rename(columns={'SMILES_y': 'SMILES', 'MolWt': 'Molecular Weight', 'BalabanJ': 'Balaban J',
                             'BertzCT': 'Complexity Index'}, inplace=True)
    # Obtain extra context to provide new interface
    new_df = final_df[["Compound ID", "SMILES", "Molecular Weight", "Complexity Index", "Solubility"]]
    new_df.columns = [col.lower() if col != "SMILES" else col for col in new_df.columns]
    # Instantiate LLM model through ask-tell interface
    bolift_at = bolift.AskTellFewShotTopk()
    qaf_at = qafnet.QAFFewShotTopK()
    # Tell the model some points (few-shot/ICL)
    mini_df = new_df.head(5)
    icl_examples = []
    icl_values = []
    # BO-Lift model
    for _, row in mini_df.iterrows():
        icl_examples.append(row["compound id"])
        icl_values.append(row["solubility"])
        bolift_at.tell(row["compound id"], row["solubility"])
    # QAFNet model
    qaf_at.tell(data=mini_df)
    # Make a prediction for a molecule
    molecule_name = new_df.iloc[6]["compound id"]
    molecule_sol = new_df.iloc[6]["solubility"]
    bolift_pred = bolift_at.predict(molecule_name)
    qaf_pred = qaf_at.predict(new_df.iloc[6][:-1])
    # Find the MAE and Standard Deviation
    if isinstance(bolift_pred, list):
        bolift_mae = np.abs(molecule_sol - bolift_pred[0].mean())
        bolift_std = bolift_pred[0].std()
    else:
        bolift_mae = np.abs(molecule_sol - bolift_pred.mean())
        bolift_std = bolift_pred.std()
    if isinstance(qaf_pred, list):
        qaf_mae = np.abs(molecule_sol - qaf_pred[0].mean())
        qaf_std = qaf_pred[0].std()
    else:
        qaf_mae = np.abs(molecule_sol - qaf_pred.mean())
        qaf_std = qaf_pred.std
    # LLM as BO (QAF)
    pool_list_qaf = [dict(row) for _, row in new_df.loc[7:10].iloc[:, :-1].iterrows()]
    # Create the pool object
    pool_qaf = qafnet.Pool(pool_list_qaf)
    # Ask the next point
    next_point_qaf = qaf_at.ask(pool_qaf)
    qaf_at.tell(new_df[new_df['compound id'] == next_point_qaf[0][0]['compound id']])
    qaf_new_bo_pred = qaf_at.predict(new_df.iloc[6][:-1])
    # LLM as BO (BO-Lift)
    pool_list_bo = list(new_df.iloc[7:11]['compound id'].values)
    pool_bo = bolift.Pool(pool_list_bo)
    next_point_bo = bolift_at.ask(pool_list_bo)
    bolift_at.tell()
    print(f"Molecule {molecule_name} has solubility level {molecule_sol}. The following algorithms return: \n")
    print(f"BO-Lift: MAE = {bolift_mae} | Standard Deviation = {bolift_std}.")
    print(f"QAF-Net: MAE = {qaf_mae} | Standard Deviation = {qaf_std}.")
