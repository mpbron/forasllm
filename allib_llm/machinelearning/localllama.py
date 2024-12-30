from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import torch
from allib.environment.base import AbstractEnvironment
from allib.instances.abstracts import PaperAbstractInstance
from langchain_community.llms import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    Pipeline,
    pipeline,
)

from .llm import LLMInterface


class LocalLlama(LLMInterface):
    model_path: Path
    pipeline: Pipeline
    tokenizer: LlamaTokenizer
    device: str
    max_sequence_length: int

    def __init__(
        self, model_path: Path, device: str = "cuda:1", max_sequence: int = 200
    ):
        self.model_path = model_path
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.pipeline = pipeline(
            "text-generation",
            model=str(self.model_path),
            torch_dtype=torch.float16,
            device_map=device,
            max_new_tokens=max_sequence,
        )
        self.max_sequence_length = max_sequence

    def run_prompt(self, prompt: str) -> Optional[str]:
        res: Sequence[Mapping[str, str]] = self.pipeline(  # type: ignore
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=self.max_sequence_length,
        )
        if res and "generated_text" in res[0]:
            return res[0]["generated_text"]
        return None

    def run_prompts(self, prompts: Sequence[str]) -> Sequence[str]:
        res: Sequence[Sequence[Mapping[str, str]]] = self.pipeline(  # type: ignore
            prompts,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=self.max_sequence_length,
        )
        return [seq[0]["generated_text"] for seq in res]


def langchain_llama(model_path: Union[Path, str], device: str = "cuda:1", max_sequence: int = 200):
    model_path = model_path
    hf_pipeline = pipeline(
        "text-generation",
        model=str(model_path),
        torch_dtype=torch.float16,
        device_map=device,
        max_new_tokens=max_sequence,
    )
    return HuggingFacePipeline(pipeline=hf_pipeline)

def gptq_pipeline(model_name_or_path: Union[Path, str], revision:str="gptq-4bit-32g-actorder_True",
                  device: str = "auto", max_sequence: int = 200, temperature=0.3):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             torch_dtype=torch.float16,
                                             device_map="auto",
                                             revision=revision)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_sequence,
        temperature=temperature,
        top_p=0.95,
        repetition_penalty=1.15
    )
    return HuggingFacePipeline(pipeline=hf_pipeline)



