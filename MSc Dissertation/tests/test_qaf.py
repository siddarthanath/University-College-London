"""
This file tests the QAF interface Vs BO-Lift interface.
"""
# -------------------------------------------------------------------------------------------------------------------- #

# Standard Library
import os

# Third Party
import numpy as np
import pandas as pd

# Private
import qafnet
import bolift
import qafnet


def test_paper():
    np.random.seed(0)
    os.environ["OPENAI_API_KEY"] = "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"
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
