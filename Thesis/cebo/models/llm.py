"""
This file creates the LLM model.
This is the main file which allows us to run the ask-tell interface and Bayesian optimisation protocol.
"""
# -------------------------------------------------------------------------------------------------------------------- #

# Standard Library
import re

# Third Party
import numpy as np
import langchain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.cache import InMemoryCache
from langchain.schema import HumanMessage, SystemMessage

# Private Party
from cebo.helper.utils import make_dd


# -------------------------------------------------------------------------------------------------------------------- #


class LLM:
    langchain.llm_cache = InMemoryCache()

    def __init__(self, model: str, temperature: float):
        self.model = model
        self.temperature = temperature

    def get_llm(
        self, n=1, top_p=1, best_of=1, max_tokens=128, logit_bias=None, **kwargs
    ):
        if logit_bias is None:
            logit_bias = {}
        if self.model in ["gpt-4", "gpt-3.5-turbo"]:
            if "logprobs" in kwargs:
                # not supported
                del kwargs["logprobs"]
            return ChatOpenAI(
                model_name=self.model,
                temperature=self.temperature,
                n=n,
                model_kwargs=kwargs,
                max_tokens=max_tokens,
            )
        else:
            return OpenAI(
                model_name=self.model,
                temperature=self.temperature,
                n=n,
                best_of=best_of,
                top_p=top_p,
                model_kwargs=kwargs,
                logit_bias=logit_bias,
                max_tokens=max_tokens,
            )

    @staticmethod
    def wrap_chatllm(query_list, llm):
        if type(llm) == ChatOpenAI:
            system_message_prompt = SystemMessage(
                content="You are a bot that can predict chemical and material properties. Do not explain answers, just provide numerical predictions."
            )
            if type(query_list) == str:
                query_list = [system_message_prompt, HumanMessage(content=query_list)]
            else:
                query_list = [
                    [system_message_prompt, HumanMessage(content=q)] for q in query_list
                ]
        return query_list

    @staticmethod
    def truncate(s):
        try:
            return re.findall(r"[-+]?\d*\.\d+|\d+", s)[0]
        except IndexError:
            return s

    @staticmethod
    def parse_response(generation, prompt, llm):
        # first parse the options into numbers
        text = generation.text
        matches = re.findall(r"([A-Z])\. .*?([\+\-\d][\d\.e]*)", text)
        values = dict()
        k = None
        for m in matches:
            try:
                k, v = m[0], float(m[1])
                values[k] = v
            except ValueError:
                pass
            k = None
        # now get log prob of tokens after Answer:
        tokens = generation.generation_info["logprobs"]["top_logprobs"]
        offsets = generation.generation_info["logprobs"]["text_offset"]
        if "Answer:" not in text:
            # try to extend
            c_generation = llm.generate([prompt + text + "\nAnswer:"]).generations[0][0]
            logprobs = c_generation.generation_info["logprobs"]["top_logprobs"][0]
        else:
            # find token probs for answer
            # feel like this is supper brittle, but not sure what else to try
            at_answer = False
            for i in range(len(offsets)):
                start = offsets[i] - offsets[0]
                end = offsets[i + 1] - offsets[0] if i < len(offsets) - 1 else -1
                selected_token = text[start:end]
                if "Answer" in selected_token:
                    at_answer = True
                if at_answer and selected_token.strip() in values:
                    break
            logprobs = tokens[i]
        result = [
            (values[k.strip()], v) for k, v in logprobs.items() if k.strip() in values
        ]
        probs = np.exp(np.array([v for k, v in result]))
        probs = probs / np.sum(probs)
        # return DiscreteDist(np.array([k for k, v in result]), probs)
        return make_dd(np.array([k for k, v in result]), probs)

    def parse_response_topk(self, generations):
        values, logprobs = [], []
        for gen in generations:
            try:
                v = float(self.truncate(gen.text))
                values.append(v)
            except ValueError:
                continue
            # can do inner sum because there is only one token
            lp = sum(
                [
                    sum(x.to_dict().values())
                    for x in gen.generation_info["logprobs"]["top_logprobs"]
                ]
            )
            logprobs.append(lp)
        probs = np.exp(np.array(logprobs))
        probs = probs / np.sum(probs)
        # return DiscreteDist(np.array(values), probs)
        return make_dd(np.array(values), probs)

    def parse_response_n(self, generations):
        values = []
        for gen in generations:
            try:
                v = float(self.truncate(gen.text))
                values.append(v)
            except ValueError:
                continue
        probs = [1 / len(values) for _ in values]
        # return DiscreteDist(np.array(values), probs)
        return make_dd(np.array(values), probs)

    def openai_choice_predict(self, query_list, llm, verbose, *args, **kwargs):
        """Predict the output numbers for a given list of queries"""
        with get_openai_callback() as cb:
            completion_response = llm.generate(query_list, *args, **kwargs)
            token_usage = cb.total_tokens
        if verbose:
            print("-" * 80)
            print(query_list[0])
            print("-" * 80)
            print(query_list[0] + completion_response.generations[0][0].text)
            print("-" * 80)
        results = []
        for gen, q in zip(completion_response.generations, query_list):
            results.append(self.parse_response(gen[0], q, llm))
        return results, token_usage

    def openai_topk_predict(self, query_list, llm, verbose, *args, **kwargs):
        """Predict the output numbers for a given list of queries"""
        query_list = self.wrap_chatllm(query_list, llm)
        with get_openai_callback() as cb:
            completion_response = llm.generate(query_list, *args, **kwargs)
            token_usage = cb.total_tokens
        if verbose:
            print("-" * 80)
            print(query_list[0])
            print("-" * 80)
            print(query_list[0] + completion_response.generations[0][0].text)
            print("-" * 80)
        results = []
        for gens in completion_response.generations:
            if type(llm) == ChatOpenAI:
                results.append(self.parse_response_n(gens))
            else:
                results.append(self.parse_response_topk(gens))
        return results, token_usage
