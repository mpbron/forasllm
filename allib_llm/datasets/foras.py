from __future__ import annotations

import operator
from functools import reduce
from pathlib import Path
from string import ascii_lowercase
from typing import (
    Any,
    Callable,
    FrozenSet,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import instancelib as il
import pandas as pd
from allib.benchmarking.reviews import read_metadata
from allib.environment.abstracts import (
    PaperAbstractEnvironment,
    text_builder,
    text_dict,
)
from allib.environment.memory import MemoryEnvironment
from allib.instances.abstracts import MemoryPaperAbstractInstanceProvider
from instancelib.ingest.spreadsheet import (
    id_col,
    instance_extractor,
    no_vector,
    text_concatenation,
)
from instancelib.labels.memory import MemoryLabelProvider
from instancelib.typehints.typevars import KT, LT
from instancelib.utils.func import identity, list_unzip3
from trinary import Trinary, Unknown, strictly, weakly

from ..machinelearning.langchain import QAResult

COLUMNS_TO_KEEP = [
    "MID",
    "title",
    "abstract",
    "title_eligible_Bruno",
    "TI-AB_IC1_Bruno",
    "TI-AB_IC2_Bruno",
    "TI-AB_IC3_Bruno",
    "TI-AB_IC4_Bruno",
    "TI-AB_other_exlusion_reason_Bruno",
    "TI-AB_final_label_Bruno",
]

NA_COLS = [
    "title",
    "abstract",
    "TI-AB_IC1_Bruno",
    "TI-AB_IC2_Bruno",
    "TI-AB_IC3_Bruno",
    "TI-AB_IC4_Bruno",
]


def clean_up(df: pd.DataFrame) -> pd.DataFrame:
    # Only keep the info that is relevant for us

    df = df[COLUMNS_TO_KEEP]

    # remove the rows of wich the title is not eligible
    df = df[df["title_eligible_Bruno"].isin(["Y", "y"])]

    # there are still 92 rows with NaN values in the columns that we are interested in. We will remove these rows
    df = df.dropna(subset=NA_COLS)
    return df


def foras_to_env(
    path: Path,
    title_col: str = "title",
    abstract_col: str = "abstract",
    label_cols: Sequence[str] = [
        "TI-AB_IC1_Bruno",
        "TI-AB_IC2_Bruno",
        "TI-AB_IC3_Bruno",
        "TI-AB_IC4_Bruno",
    ],
) -> MemoryEnvironment:
    motherfile = pd.read_excel(path)
    df = clean_up(motherfile)
    symbol_mapping = dict(zip(label_cols, ascii_lowercase))
    status_mapping = {"Y": "Y", "N": "N", "Q": "U", "M": "U"}

    def row_to_label(row: pd.Series) -> FrozenSet[str]:
        return frozenset(
            {f"{symbol_mapping[col]}_{status_mapping[row[col]]}" for col in label_cols}
        )

    triples = instance_extractor(
        df,
        id_col("MID"),
        text_dict(title_col, abstract_col),
        no_vector(),
        text_concatenation(title_col, abstract_col),
        row_to_label,
        text_builder,
    )
    keys, instances, labels = list_unzip3(triples)
    dataset = MemoryPaperAbstractInstanceProvider(instances)
    labels = MemoryLabelProvider.from_tuples(list(zip(keys, labels)))

    metadata_file = path.parent / f"{path.stem}.yaml"
    metadata = read_metadata(metadata_file)
    il_env = PaperAbstractEnvironment(dataset, labels)
    al_env = MemoryEnvironment.from_instancelib_simulation(il_env, metadata=metadata)
    return al_env

def create_key_converter(path: Path) -> Callable[[int],Optional[str]]:
    motherfile = pd.read_excel(path)
    mapping: Mapping[int, str] = dict(zip(motherfile.index, motherfile["MID"]))
    return mapping.get

def char_to_trinary(char: str) -> Trinary:
    mapping = {"Y": True, "N": False, "U": Unknown}
    return mapping.get(char, Unknown)


def symbol_label_viewer(
    labelset: Optional[Iterable[LT]] = None,
    prefix: str = "",
    boolmapper: Callable[[bool], Any] = identity,
) -> Callable[[KT, il.LabelProvider[KT, LT]], Mapping[str, Any]]:
    def viewer(key: KT, labelprovider: il.LabelProvider[KT, LT]) -> Mapping[str, Any]:
        symbols: list[Tuple[str, Trinary]] = sorted(
            [
                (elem[0], char_to_trinary(elem[1]))
                for elem in [str(lbl).split("_") for lbl in labelprovider[key]]
            ]
        )
        judgement: Trinary = reduce(
            operator.and_, [symbol for (_, symbol) in symbols], True
        )
        return {**{k: str(s) for (k, s) in symbols}, **{"judgement": str(judgement)}}

    return viewer


def answers_to_provider(anss: Sequence[QAResult]) -> il.LabelProvider[str, str]:
    tuples = [(ans.key, ans.to_labels(["inclusions"])) for ans in anss]
    provider = il.MemoryLabelProvider.from_tuples(tuples)
    return provider



