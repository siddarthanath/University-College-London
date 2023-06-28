"""
This file creates extends the original Ask-Tell interface by incorporating contextual information for solubility prediction.
This methods adapts the prefix and the prompt template, in attempt to improve prediction accuracy.
Note that there are other ways to incorporate contextual information into the LLM.
"""
# -------------------------------------------------------------------------------------------------------------------- #

# Standard Library
import re
from typing import *
from functools import partial

# Third Party
import numpy as np
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import FAISS, Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.example_selector import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)
from pydantic import BaseModel, create_model, Field, validator

# Private
from .llm_model import (
    get_llm,
    openai_choice_predict,
    openai_topk_predict,
    DiscreteDist,
    GaussDist,
    wrap_chatllm,
)
from .aqfxns import (
    probability_of_improvement,
    expected_improvement,
    upper_confidence_bound,
    greedy,
)
from .pool import Pool

# -------------------------------------------------------------------------------------------------------------------- #


# Test the function
keywords = ["molecule name", "molecule weight"]
datatypes = [str, Union[float, int]]

DynamicModel = generate_model(keywords, datatypes)
DynamicModel

class ChemicalCompound(BaseModel):
    """This creates a class that defines the properties/features of the chemical compound."""
    name: StrictStr = Field(..., description="This is name of the chemical compound i.e. IUPAC.", allow_mutation=False)
    smiles: StrictStr = Field(..., description="This is the string formula representation of the chemical compound.", allow_mutation=False)
    mol_weight: Union[StrictFloat, StrictInt] = Field(..., description="This is molecular weight of the chemical compound.", allow_mutation=False, ge=0)
    top_sa: Union[StrictFloat, StrictInt] = Field(..., description="This is the topological surface area of the chemical compound.", allow_mutation=False, ge=0)
    top_ci: Union[StrictFloat, StrictInt] = Field(..., description="This is the topological complexity index of the chemical compound.", allow_mutation=False, ge=0)
    bj_index: Union[StrictFloat, StrictInt] = Field(..., description="This is the balaban's J index of the chemical compound.", allow_mutation=False, ge=0)
    solubility: Union[float, int] = Field(..., description="This is what the LLM is going to predict.", allow_mutation=False)
    
class QAFFewShot():

    def __init__(
            self,
            prompt_template: PromptTemplate = None,
            suffix: Optional[str] = None,
            model: str = "text-curie-001",
            temperature: Optional[float] = None,
            prefix: Optional[str] = None,
            selector_k: Optional[int] = None,
            k: int = 5,
            verbose: bool = False,
            cos_sim: bool = False,
    ) -> None:
        """
        Initialize Ask-Tell Few Shot Multi optimizer.

        Args:
            prompt_template: 
                Prompt template that should take x and y (for few shot templates)
            suffix: 
                Matching suffix for first part of prompt template - for actual completion.
            model: 
                OpenAI base model to use for training and inference.
            temperature: 
                Temperature to use for inference. If None, will use model default.
            prefix: 
                Prefix to add before all examples (e.g., some context for the model).
            selector_k: 
                What k to use when switching to selection mode. If None, will use all examples.
            k: 
                Number of examples to use for each prediction.
            verbose: 
                Whether to print out debug information.
            cos_sim:
                Cosine similarity metric used in MMR.
        Returns:
            N/A
        """

        self._selector_k = selector_k
        self._ready = False
        self._ys = []
        self._prompt_template = prompt_template
        self._suffix = suffix
        self._prefix = prefix
        self._model = model
        self._example_count = 0
        self._temperature = temperature
        self._k = k
        self._answer_choices = _answer_choices[:k]
        self._calibration_factor = None
        self._verbose = verbose
        self.tokens_used = 0
        self.cos_sim = cos_sim

    def tell(self, examples: List[Dict[str, Optional[Union[float, str, int]]]]) -> None:
        """
        Tell the optimizer about new examples.

        Args:
            examples:
                This is a dictionary containing the information needed to create the prompt.
        
        """
        # Obtain model information
        example_dict, inv_example = self._tell(examples=examples)
        # If no examples have been provided in prompt, include one!
        if not self._ready:
            self.prompt = self._setup_prompt(
                example_dict, self._prompt_template, self._suffix, self._prefix
            )
            self.inv_prompt = self._setup_inverse_prompt(inv_example)
            self.llm = self._setup_llm(self._model, self._temperature)
            self.inv_llm = self._setup_inv_llm(self._model, self._temperature)
            self._ready = True
        # If one example has been provided, then check for repeats
        else:
            if self._selector_k is not None:
                self.prompt.example_selector.add_example(example_dict)
                self.inv_prompt.example_selector.add_example(inv_example)
            else:
                self.prompt.examples.append(example_dict)
                self.inv_prompt.examples.append(inv_example)
        self._example_count += 1

    """Tell the optimizer about a new example."""

    def _tell(self, examples: List[Dict[str, Optional[Union[float, str, int]]]]) -> Tuple[Dict, Dict]:
        # Make sure examples is not empty
        if len(examples) < 1:
            raise ValueError("No examples have been provided. Please give the LLM model information for your given problem.")
        # Form dictionary of necessary components to form prompt
        else:
            # Loop through each example
            for example in examples:
                # Store output values
                self._ys.append(examples["output"])

                # Store the necessary information for the problem
                model_info = ModelInfo(output=examples["output"], 
                                       )
                example_dict = 
                # Store the output values
                self._ys.append(y)
       
        return example_dict, inv_dict
    
    def _setup_prompt(
            self,
            example: Dict,
            prompt_template: Optional[PromptTemplate] = None,
            suffix: Optional[str] = None,
            prefix: Optional[str] = None,
    ) -> FewShotPromptTemplate:
        """
        This enables the creation of a prompt template, which will be passed to the LLM.
        
        Args:


        """

        # Setup prefix i.e. information for the LLM to process
        if prefix is None:
            prefix = ("You are a chemist tasked with predicting the solubility of various compounds. "
                      "You are given the SMILES representation of the compound and you need to predict its log solubility. "
                      "The following are correctly answered questions. "
                      "Each answer is numeric and ends with ###\n")

        if prompt_template is None:
            prompt_template = PromptTemplate(
                input_variables=["x", "y", "y_name"],
                template="In the field of chemistry, solubility is a key property of compounds. It can be predicted using the structure of the compound. Here, the structure of the compound is given in SMILES notation, {x}. Your task was to predict the log solubility {y_name} of the compound. For example, if the SMILES string is '{x}', the predicted log solubility is {y}.###\n\n",
            )
            if suffix is not None:
                raise ValueError(
                    "Cannot provide suffix if using default prompt template."
                )
            suffix = "In the field of chemistry, solubility is a key property of compounds. It can be predicted using the structure of the compound. Here, the structure of the compound is given in SMILES notation, {x}. What is the predicted log solubility {y_name}?"
        elif suffix is None:
            raise ValueError("Must provide suffix if using custom prompt template.")
        # Test prompt
        if example is not None:
            prompt_template.format(**example)
            examples = [example]
        # Use MMR to select the best examples (if cosine similarity is passed)
        example_selector = None
        if self._selector_k is not None:
            if len(examples) == 0:
                raise ValueError("Cannot do zero-shot with selector")
            if not self.cos_sim:
                example_selector = (
                    example_selector
                ) = MaxMarginalRelevanceExampleSelector.from_examples(
                    [example],
                    OpenAIEmbeddings(),
                    FAISS,
                    k=self._selector_k,
                )
            else:
                example_selector = (
                    example_selector
                ) = SemanticSimilarityExampleSelector.from_examples(
                    [example],
                    OpenAIEmbeddings(),
                    Chroma,
                    k=self._selector_k,
                )
        return FewShotPromptTemplate(
            examples=examples if example_selector is None else None,
            example_prompt=prompt_template,
            example_selector=example_selector,
            suffix=suffix,
            prefix=prefix,
            input_variables=["x", "y_name"],
        )

    """This enables the creation of an LLM from OpenAI."""

    def _setup_llm(self, model: str, temperature: Optional[float] = None):
        # Return LLM
        return get_llm(
            n=self._k,
            best_of=self._k,
            temperature=0.1 if temperature is None else temperature,
            model_name=model,
            top_p=1.0,
            stop=["\n", "###", "#", "##"],
            logit_bias={
                "198": -100,
                "628": -100,
                "50256": -100,
            },
            max_tokens=256,
            logprobs=1,
        )
    
    """This enables the creation of an inverse prompt template, which will be passed to the LLM."""

    def _setup_inverse_prompt(self, example: Dict):
        prompt_template = PromptTemplate(
            input_variables=["x", "y", "y_name", "x_name"],
            template="If solubility {y_name} is {y}, then {x_name} has SMILES representation {x}\n\n",
        )
        if example is not None:
            prompt_template.format(**example)
            examples = [example]
        else:
            examples = []
        example_selector = None
        if self._selector_k is not None:
            if len(examples) == 0:
                raise ValueError("Cannot do zero-shot with selector")
            if not self.cos_sim:
                example_selector = (
                    example_selector
                ) = MaxMarginalRelevanceExampleSelector.from_examples(
                    [example],
                    OpenAIEmbeddings(),
                    FAISS,
                    k=self._selector_k,
                )
            else:
                example_selector = (
                    example_selector
                ) = SemanticSimilarityExampleSelector.from_examples(
                    [example],
                    OpenAIEmbeddings(),
                    Chroma,
                    k=self._selector_k,
                )
        return FewShotPromptTemplate(
            examples=examples if example_selector is None else None,
            example_prompt=prompt_template,
            example_selector=example_selector,
            suffix="If solubility {y_name} is {y}, then {x_name} has SMILES representation ",
            input_variables=["y", "y_name", "x_name"],
        )



    """Using the LLM, return predictions."""

    def _predict(self, queries: List[str]) -> List[DiscreteDist]:
        # Call OpenAI internal prediction
        result, token_usage = openai_topk_predict(queries, self.llm, self._verbose)
        # If quantiles has been provided without using quantile transformer
        if self.use_quantiles and self.qt is None:
            raise ValueError(
                "Can't use quantiles without building the quantile transformer"
            )
        if self.use_quantiles:
            for r in result:
                if isinstance(r, GaussDist):
                    r._mean = self.qt.to_values(r._mean)
                elif isinstance(r, DiscreteDist):
                    r.values = self.qt.to_values(r.values)
        return result, token_usage
    

    def ask(
            self,
            possible_x: Union[Pool, List[str]],
            aq_fxn: str = "upper_confidence_bound",
            k: int = 1,
            inv_filter: int = 16,
            _lambda: float = 0.5,
    ) -> Tuple[List[str], List[float], List[float]]:
        """Ask the optimizer for the next x to try.
        Args:
            possible_x: List of possible x values to choose from.
            aq_fxn: Acquisition function to use.
            k: Number of x values to return.
            inv_filter: Reduce pool size to this number with inverse model. If 0, not used
            _lambda: Lambda value to use for UCB
            context_vales: The values from the smiles embeddings.
        Return:
            The selected x values, their acquisition function values, and the predicted y modes.
            Sorted by acquisition function value (descending)
        """
        # Initialise pool for BO
        if type(possible_x) == type([]):
            possible_x = Pool(possible_x, self.format_x)
        # Pick acquisition function
        if aq_fxn == "probability_of_improvement":
            aq_fxn = probability_of_improvement
        elif aq_fxn == "expected_improvement":
            aq_fxn = expected_improvement
        elif aq_fxn == "upper_confidence_bound":
            aq_fxn = partial(upper_confidence_bound, _lambda=_lambda)
        elif aq_fxn == "greedy":
            aq_fxn = greedy
        elif aq_fxn == "random":
            return (
                possible_x.sample(k),
                [0] * k,
                [0] * k,
            )
        else:
            raise ValueError(f"Unknown acquisition function: {aq_fxn}")
        # Choose current best target value
        if len(self._ys) == 0:
            best = 0
        else:
            best = np.max(self._ys)
        # If inverse filter is provided, sample!
        if inv_filter != 0 and inv_filter < len(possible_x):
            approx_x = self.inv_predict(best * np.random.normal(1.0, 0.05))
            possible_x_l = possible_x.approx_sample(approx_x, inv_filter)
        else:
            possible_x_l = list(possible_x)
        results = self._ask(possible_x_l, best, aq_fxn, k)
        # If we have no results, return a random one
        if len(results[0]) == 0 and len(possible_x_l) != 0:
            return (
                possible_x.sample(k),
                [0] * k,
                [0] * k,
            )
        return results

    def _ask(
            self, possible_x: List[str], best: float, aq_fxn: Callable, k: int
    ) -> Tuple[List[str], List[float], List[float]]:
        # Use LLM predict to obtain results from possible x values in pool
        results = self.predict(possible_x)
        # Drop empty results
        results = [r for r in results if len(r) > 0]
        aq_vals = [aq_fxn(r, best) for r in results]
        selected = np.argsort(aq_vals)[::-1][:k]
        means = [r.mean() for r in results]
        # Return values with their acquisition scores and mean
        return (
            [possible_x[i] for i in selected],
            [aq_vals[i] for i in selected],
            [means[i] for i in selected],
        )
    

    def predict(self, x: str) -> Union[Tuple[float, float], List[Tuple[float, float]]]:
        """Predict the probability distribution and values for a given x.

        Args:
            x: Input information.
        Returns:
            The probability distribution and values for the given x.

        """
        # Make input into a list
        if not isinstance(x, list):
            x = [x]
        if not self._ready:
            # Zero-Shot Learning
            self.prompt = self._setup_prompt(
                None, self._prompt_template, self._suffix, self._prefix
            )
            self.inv_prompt = self._setup_inverse_prompt(None)
            self.llm = self._setup_llm(self._model)
            self._ready = True
        # NOTE: Needs updating!
        if self._selector_k is not None:
            self.prompt.example_selector.k = min(self._example_count, self._selector_k)
        # Generate queries from prompts
        queries = [
            self.prompt.format(
                x=self.format_x(x_i),
                y_name=self._y_name,
            )
            for x_i in x
        ]
        results, tokens = self._predict(queries)
        self.tokens_used += tokens
        # Replace Gauss Dist with population std!
        for i, result in enumerate(results):
            if len(self._ys) > 1:
                ystd = np.std(self._ys)
            elif len(self._ys) == 1:
                ystd = self._ys[0]
            else:
                ystd = 10
            if isinstance(result, GaussDist):
                results[i].set_std(ystd)
        # Adapt results if calibration factor is used
        if self._calibration_factor:
            for i, result in enumerate(results):
                if isinstance(result, GaussDist):
                    results[i].set_std(result.std() * self._calibration_factor)
                elif isinstance(result, DiscreteDist):
                    results[i] = GaussDist(
                        results[i].mean(),
                        results[i].std() * self._calibration_factor,
                    )
        # Compute mean and standard deviation
        if len(x) == 1:
            return results[0]
        return results
    
  