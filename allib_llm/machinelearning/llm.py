from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Mapping, Optional, Sequence

import numpy as np
import numpy.typing as npt
import requests
from allib.environment.base import AbstractEnvironment
from allib.typehints.typevars import DT, IT, KT, LT, RT, VT
from environs import Env
from typing_extensions import Self

# For local streaming, the websockets are hosted without ssl - http://

DEFAULT_PARAMS_LLM = {
    "max_new_tokens": 250,
    # Generation params. If 'preset' is set to different than 'None', the values
    # in presets/preset-name.yaml are used instead of the individual numbers.
    "preset": "Divine Intellect",
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.1,
    "typical_p": 1,
    "epsilon_cutoff": 0,  # In units of 1e-4
    "eta_cutoff": 0,  # In units of 1e-4
    "tfs": 1,
    "top_a": 0,
    "repetition_penalty": 1.18,
    "repetition_penalty_range": 0,
    "top_k": 40,
    "min_length": 0,
    "no_repeat_ngram_size": 0,
    "num_beams": 1,
    "penalty_alpha": 0,
    "length_penalty": 1,
    "early_stopping": False,
    "mirostat_mode": 0,
    "mirostat_tau": 5,
    "mirostat_eta": 0.1,
    "seed": -1,
    "add_bos_token": True,
    "truncation_length": 2048,
    "ban_eos_token": False,
    "skip_special_tokens": True,
    "stopping_strings": [],
}


def default_pretext(pos_label: LT, neg_label: LT) -> str:
    return """You are an assistant evaluating whether or not an abstract should be included into a systematic review based on criteria stated below:"""


def default_instructions(pos_label: LT, neg_label: LT) -> str:
    return f"Write EITHER [{pos_label}] OR [{neg_label}] based on the criteria. Write nothing else."


def longer_pretext(pos_label: LT, neg_label: LT) -> str:
    return "You are a researcher rigorously screening titles and abstracts of scientific papers for inclusion or exclusion in a review paper. Use the criteria below to inform your decision. If any exclusion criteria are met or not all inclusion criteria are met, exclude the article. If all inclusion criteria are met, include the article."


def explain_first(pos_label: LT, neg_label: LT) -> str:
    return f"Give a brief explanation of your decision based on the criteria. Finish your answer with EITHER [{pos_label}] OR [{neg_label}]."


def answer_first_then_explain(pos_label: LT, neg_label: LT) -> str:
    return f"Write EITHER [{pos_label}] OR [{neg_label}] based on the criteria. Then explain why."


class LLMInterface(ABC):
    @abstractmethod
    def run_prompt(self, prompt: str) -> Optional[str]:
        raise NotImplementedError


class LLMAPI(LLMInterface):
    endpoint_uri: str
    params: Mapping[str, Any]

    def __init__(
        self, endpoint_uri: str, params: Mapping[str, Any] = DEFAULT_PARAMS_LLM
    ) -> None:
        super().__init__()
        self.endpoint_uri = endpoint_uri
        self.params = params

    def run_prompt(self, prompt: str) -> Optional[str]:
        request = {"prompt": prompt, **self.params}
        response = requests.post(self.endpoint_uri, json=request)

        if response.status_code == 200:
            result = response.json()["results"][0]["text"]
            return result
        return None


class LLMClassifier(Generic[DT, LT]):
    llm: LLMInterface
    inclusion_criteria: str
    exclusion_criteria: str
    endpoint_uri: str
    pretext_generator: Callable[[LT, LT], str]
    instruction_generator: Callable[[LT, LT], str]
    pos_label: LT
    neg_label: LT
    labels: Sequence[LT]

    def __init__(
        self,
        llm: LLMInterface,
        inclusion_criteria: str,
        exclusion_criteria: str,
        pos_label: LT,
        neg_label: LT,
        pretext_generator: Callable[[LT, LT], str] = default_pretext,
        instruction_generator: Callable[[LT, LT], str] = default_instructions,
    ):
        self.llm = llm
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.labels = (neg_label, pos_label)
        self.pretext_generator = pretext_generator
        self.instruction_generator = instruction_generator
        self.inclusion_criteria = inclusion_criteria
        self.exclusion_criteria = exclusion_criteria

    def data_formatter(self, data: DT) -> str:
        return str(data)

    def data_to_prompt(self, data: DT) -> str:
        pretext = self.pretext_generator(self.pos_label, self.neg_label)
        instructions = self.instruction_generator(self.pos_label, self.neg_label)
        data_formatted = self.data_formatter(data)
        prompt = (
            "## HUMAN:\n"
            f"SITUATION:\n {pretext} \n"
            f"INCLUSION CRITERIA:\n{self.inclusion_criteria}\n"
            f"EXCLUSION CRITERIA:\n{self.exclusion_criteria}\n"
            f"{data_formatted}"
            f"INSTRUCTIONS:\n {instructions} \n"
            "## ASSISTANT:\n DECISION: \n"
        )
        return prompt

    def __call__(self, datas: Sequence[DT]) -> npt.NDArray[np.float64]:
        results = np.zeros((len(datas), 2))
        for i, data in enumerate(datas):
            assert isinstance(data, str)
            prompt = self.data_to_prompt(data)
            response = self.llm.run_prompt(prompt)
            if response is not None:
                for j, label in enumerate(self.labels):
                    if response == label:
                        results[i, j] = 1.0
        return results

    @classmethod
    def build(
        cls,
        env: AbstractEnvironment[Any, Any, DT, Any, Any, LT],
        llm: LLMInterface,
        pos_label: LT,
        neg_label: LT,
        pretext_generator: Callable[[LT, LT], str] = default_pretext,
        instruction_generator: Callable[[LT, LT], str] = default_instructions,
    ) -> Self:
        classifier = cls(
            llm,
            env.metadata["inclusion_criteria"],
            env.metadata["exclusion_criteria"],
            pos_label,
            neg_label,
            pretext_generator=pretext_generator,
            instruction_generator=instruction_generator,
        )
        return classifier


class TitleAbstractLLM(LLMClassifier[Mapping[str, str], LT], Generic[LT]):
    def data_formatter(self, data: Mapping[str, str]) -> str:
        title = data["title"]
        abstract = data["abstract"]
        return f"TITLE: {title}\nABSTRACT: {abstract}\n"
