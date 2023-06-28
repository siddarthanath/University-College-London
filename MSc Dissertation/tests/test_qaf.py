"""
This file tests the QAF interface.
"""
# -------------------------------------------------------------------------------------------------------------------- #

# Standard Library
import os

# Third Party
import pytest
import numpy as np

# Private
import qafnet
import bolift
from qafnet import llm_model


np.random.seed(0)
os.environ["OPENAI_API_KEY"] = "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"

def test_paper_testing():
    # Instantiate LLM model through ask-tell interface
    asktell = bolift.AskTellFewShotTopk()
    # Tell some points to the model
    asktell.tell("1-bromopropane", -1.730)
    asktell.tell("1-bromopentane", -3.080)
    asktell.tell("1-bromooctane", -5.060)
    asktell.tell("1-bromonaphthalene", -4.35)

    # Make a prediction
    yhat = asktell.predict("1-bromobutane")
    print(yhat.mean(), yhat.std())
    # Now treat LLM model as a BO protcol
    pool_list = [
        "1-bromoheptane",
        "1-bromohexane",
        "1-bromo-2-methylpropane",
        "butan-1-ol"
    ]
    # Create the pool object
    pool = bolift.Pool(pool_list)
    # Ask for the next most likely point (found through using UCB as the acquisition function on the previous points)
    next_point = asktell.ask(pool)
    print(f"The next point for the optimiser to try is: {next_point}")
    # Tell the model some points (few-shot/ICL)
    # Make a prediction for a molecule
    print(f"Y_Hat for ICL (before BO): {yhat}")
    print(f"Y_Hat Mean: {yhat.mean()}")
    print(f"Y_Hat Standard Deviation: {yhat.std()}")

def test_completion():
    llm = llm_model.get_llm(stop=["\n\n"])
    assert llm("The value of 1 + 1 is") == " 2"
