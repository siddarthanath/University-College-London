"""
This file creates extends the original Ask-Tell interface by incorporating contextual information for solubility
prediction.
This method adapts the prefix and the prompt template, in attempt to improve prediction accuracy.
Note that there are other ways to incorporate contextual information into the LLM.
"""
# -------------------------------------------------------------------------------------------------------------------- #

# Standard Library
from typing import *
from functools import partial

# Third Party
import numpy as np
import pandas as pd
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

# Private
from .llm_model import (
    get_llm,
    openai_topk_predict,
    DiscreteDist,
    GaussDist,
)
from .helper import (
    probability_of_improvement,
    expected_improvement,
    upper_confidence_bound,
    greedy,
)
from .pool import Pool

# -------------------------------------------------------------------------------------------------------------------- #


class QAFFewShotTopK:
    def __init__(
        self,
        prompt_template: Optional[PromptTemplate] = None,
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
        self._calibration_factor = None
        self._verbose = verbose
        self.prompt = None
        self.llm = None
        self.tokens_used = 0
        self.cos_sim = cos_sim

    def tell(self, data: pd.DataFrame):
        """
        Tell the optimizer about new examples.

        Args:
            data:
                A dataframe containing information about problem.
                NOTE: The target name should be the final column!
        Returns:
            N/A
        """
        # Obtain model information
        examples = self._tell(data=data)
        # Create model prompt
        self.prompt = self._setup_prompt(
            examples, self._prompt_template, self._suffix, self._prefix
        )
        # Create LLM
        self.llm = self._setup_llm(self._model, self._temperature)
        # Add examples to prompt
        self._example_count += len(examples)
        self._ready = True

    def _tell(self, data: pd.DataFrame) -> List[Dict]:
        """
        Tell the optimizer about examples in the dataset.

        Args:
            data:
                A dataframe containing information about problem.
        Returns:
            A list of dictionary containing the examples
        """
        # Store examples
        examples = []
        # Make sure examples is not empty
        if data.shape[0] < 1:
            raise ValueError(
                "No examples have been provided. Please give the LLM model information for your given "
                "problem."
            )
        # Form dictionary of necessary components to form prompt
        else:
            # Loop through each example
            for _, example in data.iterrows():
                # Create dictionary of context
                examples.append(dict(example))
                # Store output values
                self._ys.append(example.values[-1])
        return examples

    def _setup_prompt(
        self,
        examples: List[Dict],
        prompt_template: Optional[PromptTemplate] = None,
        suffix: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> FewShotPromptTemplate:
        """
        This enables the creation of a prompt template, which will be passed to the LLM.
        """
        # Create input variables and template
        example = examples[0]
        input_variables = list(example.keys())
        template = (
            "Q:Given "
            + ", ".join([f"{var} {{{var}}}" for var in input_variables[:-1]])
            + ", what is the "
            + f"{input_variables[-1]}?\nA: {{{input_variables[-1]}}}###\n\n "
        )

        # Setup prefix i.e. the background on the task that the LLM will perform
        if prefix is None:
            prefix = (
                "The following are correctly answered questions. "
                "Each answer is numeric and ends with ###\n"
            )
        # Setup prompt template i.e. the information the LLM will process for the given problem
        if prompt_template is None:
            prompt_template = PromptTemplate(
                input_variables=input_variables, template=template
            )
            if suffix is not None:
                raise ValueError(
                    "Cannot provide suffix if using default prompt template."
                )
            suffix = (
                "Q:Given "
                + ", ".join([f"{var} {{{var}}}" for var in input_variables[:-1]])
                + ", what is the "
                + f"{input_variables[-1]}?\nA: "
            )
        elif suffix is None:
            raise ValueError("Must provide suffix if using custom prompt template.")
        return FewShotPromptTemplate(
            examples=examples,
            example_prompt=prompt_template,
            suffix=suffix,
            prefix=prefix,
            input_variables=input_variables[:-1],
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

    def predict(
        self, data: Union[pd.DataFrame, pd.Series]
    ) -> Union[Tuple[float, float], List[Tuple[float, float]]]:
        """Predict the probability distribution and values for a given input.

        Args:
            data:
                Input information.
        Returns:
            The probability distribution and values for the given x.

        """
        # Generate queries from prompts
        if data.ndim == 1:
            queries = [self.prompt.format(**dict(data))]
        else:
            queries = [self.prompt.format(**dict(row)) for _, row in data.iterrows()]
        # Obtain results and tokens
        results, tokens = openai_topk_predict(queries, self.llm, self._verbose)
        # Store number of tokens used
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
        return results

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
