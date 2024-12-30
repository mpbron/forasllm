# %%
from __future__ import annotations

import argparse
import asyncio
import pickle
import random
from enum import Enum, auto
from pathlib import Path
from string import ascii_lowercase
from typing import Callable, FrozenSet, Mapping, Optional, Sequence, TypeVar, Union

import instancelib as il
import pandas as pd
from allib.benchmarking.reviews import read_metadata, read_review_dataset_new
from allib.environment.abstracts import (
    PaperAbstractEnvironment,
    text_builder,
    text_dict,
)
from allib.environment.base import AbstractEnvironment
from allib.environment.memory import MemoryEnvironment
from allib.instances.abstracts import MemoryPaperAbstractInstanceProvider
from environs import Env
from instancelib.ingest.spreadsheet import (
    id_index,
    instance_extractor,
    no_vector,
    text_concatenation,
)
from instancelib.labels.memory import MemoryLabelProvider
from instancelib.utils.chunks import divide_sequence
from instancelib.utils.func import list_unzip3
from langchain.base_language import BaseLanguageModel
from langchain_community.llms import HuggingFacePipeline
from langchain_openai.chat_models import ChatOpenAI
from tqdm import tqdm

from allib_llm.datasets.foras import foras_to_env
from allib_llm.machinelearning.chatgpt import create_chatgpt
from allib_llm.machinelearning.langchain import QuestionChain
from allib_llm.machinelearning.localllama import gptq_pipeline


# %%
# %%
def get_subset_by_labels_intersection(
    env: il.Environment,
    provider: il.InstanceProvider,
    *labels: str,
    labelprovider: Optional[il.LabelProvider] = None,
) -> il.InstanceProvider:
    if labelprovider is None:
        l_provider = env.labels
    else:
        l_provider = labelprovider
    keys = frozenset(provider).intersection(
        *(l_provider.get_instances_by_label(label) for label in labels)
    )
    provider = env.create_bucket(keys)
    return provider




# %%
POS = "Relevant"
NEG = "Irrelevant"

T = TypeVar("T")

env = Env()


class GPTModel(str, Enum):
    GPT35 = "gpt35"
    GPT4 = "gpt4"
    GPTQ = "gptq"


class ChainVersion(str, Enum):
    STRICT = auto()
    LENIENT = auto()
    STEPBYSTEP = auto()
    ONESTEP = auto()


def p(f: Callable[..., T], *args, **kwargs) -> Callable[[], T]:
    def g():
        return f(*args, **kwargs)
    return g


def create_gptq() -> Callable[[], HuggingFacePipeline]:
    def g():
        with env.prefixed(f"GPTQ_"):
            modelname = env("MODEL")
            branch = env("BRANCH")
            llm = gptq_pipeline(
                model_name_or_path=modelname,
                revision=branch,
                max_sequence=env.int("MAXSEQ"),
                temperature=env.float("TEMPERATURE"),
            )
            print(f"Loaded {modelname} - {branch}")
            return llm
    return g


MODELS: Mapping[GPTModel, Callable[[], Union[ChatOpenAI, BaseLanguageModel]]] = {
    GPTModel.GPT35: p(create_chatgpt, "GPT35"),
    GPTModel.GPT4: p(create_chatgpt, "GPT4"),
    GPTModel.GPTQ: create_gptq(),
}

def run_classifier(
    langchain: QuestionChain,
    ds_env: AbstractEnvironment,
    provider: il.InstanceProvider,
    path: Path,
    results_path: Path,
    pos_label: str,
    neg_label: str,
    test_max_it: Optional[int] = None,
    batch_size: int = 20,
    async_mode: bool = True,
    metadata_keys = ("inclusions", "exclusions")
):
    ds_name = path.stem
    docs = list(provider.get_all())
    df_ress = list()
    obj_ress = list()
    ds_path = results_path / ds_name
    ds_path.mkdir(parents=True, exist_ok=True)
    chunk_dir = ds_path / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for i, chunk in tqdm(list(enumerate(divide_sequence(docs, batch_size)))):
        chunk_path = chunk_dir / f"chunk_{i}.pkl"
        chunk_path_xl = chunk_dir / f"chunk_{i}.xlsx"
        chunk_path_obj = chunk_dir / f"chunk_obj_{i}.pkl"
        if not chunk_path.exists():
            obj_res, df_res = asyncio.run(langchain.arun_questions_ds(ds_env, chunk, metadata_keys))
            with chunk_path_obj.open("wb") as fh:
                pickle.dump(obj_res, fh)
            df_res.to_pickle(chunk_path)
            df_res.to_excel(chunk_path_xl, engine="xlsxwriter")
        else:
            with chunk_path_obj.open("rb") as fh:
                obj_res = pickle.load(fh)
            df_res = pd.read_pickle(chunk_path)
        df_ress.append(df_res)
        obj_ress.extend(obj_res)
    final_path = ds_path / "results.pkl"
    final_obj_path = ds_path / "results_obj.pkl"
    final_path_xl = ds_path / "results.xlsx"
    final_df = pd.concat(df_ress)
    final_df.to_pickle(final_path)
    final_df.to_excel(final_path_xl, engine="xlsxwriter")
    with final_obj_path.open("wb") as fh:
        pickle.dump(obj_ress, fh)
    return final_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="allib-llm",
        description="Active Learning Library (allib) - Large Language Model TAR classifier",
    )
    parser.add_argument("-b", "--batchsize", help="Batch Size", default=2, type=int)
    parser.add_argument("-d", "--dataset", help="The path to the dataset", type=Path)
    parser.add_argument(
        "-e", "--envfile", help="Path to the envfile", type=Path, default=Path(".env")
    )
    parser.add_argument(
        "-m", "--model", help="Choice of model", type=GPTModel, default=GPTModel.GPT35
    )
    parser.add_argument("-t", "--target", help="The target of the results", type=Path)
    parser.add_argument(
        "-p",
        "--pos_label",
        metavar="POS",
        default=POS,
        help="The label that denotes the positive class",
    )
    parser.add_argument(
        "--prompt",
        metavar="",
        type=Path,
        default=Path("./prompts/question1.txt"),
        help="The label that denotes the positive class",
    )
    parser.add_argument(
        "-n",
        "--neg_label",
        metavar="NEG",
        default=NEG,
        help="The label that denotes the negative class",
    )
    parser.add_argument(
        "-i",
        "--testiterations",
        metavar="testiterations",
        default=None,
        type=int,
        help="The number of iterations that is used to test the algorithm",
    )
    parser.add_argument("--sync", action='store_true')
    args = parser.parse_args()
    ds_path: Path = args.dataset
    target_path: Path = args.target
    pos_label: str = args.pos_label
    neg_label: str = args.neg_label
    model: GPTModel = args.model
    envfile: Path = args.envfile
    test_it = args.testiterations
    batch_size: int = args.batchsize
    asyncmode = not args.sync
    env.read_env(str(envfile), recurse=False)
    llm = MODELS[model]()
    prompt_path: Path = args.prompt
    prompt = prompt_path.read_text()
    chain = QuestionChain(llm, prompt, pos_label, neg_label)
    save_path = target_path / str(model) / "questions"
    accept_labels = frozenset({"a_Y", "b_Y", "c_Y", "d_Y"})
    question_labels = frozenset({"a_Q", "b_Q", "c_Q", "d_Q"})
    reject_labels = frozenset({"a_N", "b_N", "c_N", "d_N"})

    ds_env = foras_to_env(ds_path)
    direct_accept = get_subset_by_labels_intersection(ds_env, ds_env.dataset, *accept_labels, labelprovider=ds_env.truth)

    probable = ds_env.create_bucket(
        frozenset(ds_env.dataset).difference(
            ds_env.get_subset_by_labels(ds_env.dataset, *reject_labels, labelprovider=ds_env.truth)
        )
    )
    run_classifier(
        chain, ds_env, ds_env.dataset, ds_path, save_path, pos_label, neg_label, test_it, batch_size, asyncmode
    )
