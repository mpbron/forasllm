from __future__ import annotations

import abc
import itertools
import operator
from functools import reduce
from itertools import product
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Generic,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import instancelib as il
import pandas as pd
from allib.typehints.typevars import DT, IT, KT, LT, RT, VT
from instancelib.utils.func import union, value_map
from trinary import Trinary, Unknown, strictly, weakly

from allib_llm.datasets.foras import char_to_trinary
from allib_llm.machinelearning.langchain import trinary_to_label

_T = TypeVar("_T")


def get_subset_by_labels_intersection(
    env: il.Environment[IT, KT, DT, VT, RT, LT],
    provider: il.InstanceProvider[IT, KT, DT, VT, RT],
    *labels: LT,
    labelprovider: Optional[il.LabelProvider[KT, LT]] = None,
) -> il.InstanceProvider[IT, KT, DT, VT, RT]:
    """Get the labels from in InstanceProvider that have all the labels specified in labels

    Args:
        env (il.Environment): The Environment object of the dataset
        provider (il.InstanceProvider): The InstanceProvider that may contain Instances that have the labels
        *labels (str): Give 1 or multiple labels for which you want to find the Instances that have these labels
        labelprovider (Optional[il.LabelProvider], optional): The LabelProvider. Defaults to None, and if so, will
        use the LabelProvider within the Environment object.

    Returns:
        il.InstanceProvider: An InstanceProvider with the Instances that have all the labels specified in the arguments
    """
    if labelprovider is None:
        l_provider = env.labels
    else:
        l_provider = labelprovider
    keys = frozenset(provider).intersection(
        *(l_provider.get_instances_by_label(label) for label in labels)
    )
    ret_provider = env.create_bucket(keys)
    return ret_provider

def get_instances_with_no_labels(
    env: il.Environment[IT, KT, DT, VT, RT, LT],
    provider: il.InstanceProvider[IT, KT, DT, VT, RT],
    labelprovider: Optional[il.LabelProvider[KT, LT]] = None,
) -> il.InstanceProvider[IT, KT, DT, VT, RT]:
    """Get the labels from in InstanceProvider that have all the labels specified in labels

    Args:
        env (il.Environment): The Environment object of the dataset
        provider (il.InstanceProvider): The InstanceProvider that may contain Instances that have the labels
        *labels (str): Give 1 or multiple labels for which you want to find the Instances that have these labels
        labelprovider (Optional[il.LabelProvider], optional): The LabelProvider. Defaults to None, and if so, will
        use the LabelProvider within the Environment object.

    Returns:
        il.InstanceProvider: An InstanceProvider with the Instances that have all the labels specified in the arguments
    """
    if labelprovider is None:
        l_provider = env.labels
    else:
        l_provider = labelprovider
    with_labels = union(*(l_provider.get_instances_by_label(lbl) for lbl in l_provider.labelset))
    ret_provider = env.create_bucket(frozenset(provider).difference(with_labels))
    return ret_provider

def get_instances_with_labels(
    env: il.Environment[IT, KT, DT, VT, RT, LT],
    provider: il.InstanceProvider[IT, KT, DT, VT, RT],
    labelprovider: Optional[il.LabelProvider[KT, LT]] = None,
) -> il.InstanceProvider[IT, KT, DT, VT, RT]:
    """Get the labels from in InstanceProvider that have all the labels specified in labels

    Args:
        env (il.Environment): The Environment object of the dataset
        provider (il.InstanceProvider): The InstanceProvider that may contain Instances that have the labels
        *labels (str): Give 1 or multiple labels for which you want to find the Instances that have these labels
        labelprovider (Optional[il.LabelProvider], optional): The LabelProvider. Defaults to None, and if so, will
        use the LabelProvider within the Environment object.

    Returns:
        il.InstanceProvider: An InstanceProvider with the Instances that have all the labels specified in the arguments
    """
    if labelprovider is None:
        l_provider = env.labels
    else:
        l_provider = labelprovider
    with_labels = union(*(l_provider.get_instances_by_label(lbl) for lbl in l_provider.labelset))
    ret_provider = env.create_bucket(with_labels)
    return ret_provider


def conjunction_judgement(
    symbols: Mapping[str, Trinary], exclude: FrozenSet[str] = frozenset()
):
    judgement: Trinary = reduce(
        operator.and_,
        [status for (key, status) in symbols.items() if key not in exclude],
        True,
    )
    return judgement


def conjunction_judgement_weakly(
    symbols: Mapping[str, Trinary], exclude: FrozenSet[str] = frozenset()
):
    judgement: Trinary = reduce(
        operator.and_,
        [weakly(status) for (key, status) in symbols.items() if key not in exclude],
        True,
    )
    return judgement


def conjunction_judgement_strictly(
    symbols: Mapping[str, Trinary], exclude: FrozenSet[str] = frozenset()
):
    judgement: Trinary = reduce(
        operator.and_,
        [strictly(status) for (key, status) in symbols.items() if key not in exclude],
        True,
    )
    return judgement


def string_to_symbol(string: str) -> Trinary:
    tmap = {
        "True": True,
        "False": False,
        "Unknown": Unknown,
    }
    return tmap.get(string, Unknown)

def labels_to_symbols(lbls: FrozenSet[Any]) -> Mapping[str, Trinary]:
    symbols: Mapping[str, Trinary] = {
        elem[0]: char_to_trinary(elem[1])
        for elem in sorted([str(lbl).split("_") for lbl in lbls])
    }
    return symbols

def symbols_to_labels(trinary_map: Mapping[str, Trinary]) -> FrozenSet[str]:
    def ret():
        for qkey, val in trinary_map.items():
            status_str = trinary_to_label(val)
            yield f"{qkey}_{status_str}"
    return frozenset(ret())



class LabelTransformer(Generic[KT, LT]):
    judgement_function: Callable[
        [Mapping[str, Trinary]], Trinary
    ]

    def __init__(self, judgement_function: Callable[
        [Mapping[str, Trinary]], Trinary
    ] = conjunction_judgement) -> None:
        self.judgement_function = judgement_function

    def __call__(self, prov: il.LabelProvider[KT, Any], pos_label: LT, neg_label: LT) -> il.LabelProvider[KT, LT]:
        def trinary_to_label(val: Trinary) -> LT:
            if weakly(val):
                return pos_label
            return neg_label
        tuples = [
            (idx, frozenset([trinary_to_label(self.judgement_function(labels_to_symbols(lbls)))]))
            for idx, lbls in prov.items()
        ]
        new_prov = il.MemoryLabelProvider.from_tuples(tuples)
        return new_prov
    

class AbstractLabelTransformer(abc.ABC, Generic[KT, _T, LT]):
    @abc.abstractmethod
    def __call__(self, prov: il.LabelProvider[KT, _T]) -> il.LabelProvider[KT, LT]:
        raise NotImplementedError
    
class CriteriaToBinary(AbstractLabelTransformer[KT, Any, LT], Generic[KT, LT]):
    judgement_function: Callable[
        [Mapping[str, Trinary]], Trinary
    ]
    pos_label: LT
    neg_label: LT
    def __init__(self, 
                 pos_label: LT,
                 neg_label: LT,
                 judgement_function: Callable[[Mapping[str, Trinary]], Trinary] = conjunction_judgement) -> None:
        self.judgement_function = judgement_function
        self.pos_label = pos_label
        self.neg_label = neg_label

    def __call__(self, prov: il.LabelProvider[KT, Any]) -> il.LabelProvider[KT, LT]:
        def trinary_to_label(val: Trinary) -> LT:
            if weakly(val):
                return self.pos_label
            return self.neg_label
        tuples = [
            (idx, frozenset([trinary_to_label(self.judgement_function(labels_to_symbols(lbls)))]))
            for idx, lbls in prov.items()
        ]
        if tuples:
            new_prov = il.MemoryLabelProvider.from_tuples(tuples)
        else:
            new_prov = il.MemoryLabelProvider([self.neg_label, self.pos_label], dict(), dict())
        return new_prov


    

def labels_to_judgement_labels(
    judgement_function: Callable[
        [Mapping[str, Trinary]], Trinary
    ] = conjunction_judgement
) -> Callable[[il.LabelProvider[KT, str], LT, LT], il.LabelProvider[KT, LT]]:
    def transform(prov: il.LabelProvider[KT, str], pos_label: LT, neg_label: LT) -> il.LabelProvider[KT, LT]:
        def trinary_to_label(val: Trinary) -> LT:
            if weakly(val):
                return pos_label
            return neg_label
        tuples = [
            (idx, frozenset([trinary_to_label(judgement_function(labels_to_symbols(lbls)))]))
            for idx, lbls in prov.items()
        ]
        new_prov = il.MemoryLabelProvider.from_tuples(tuples)
        return new_prov
    return transform

def provider_to_subproviders(
    prov: il.LabelProvider[KT, str],
    judgement_function: Callable[
        [Mapping[str, Trinary]], Trinary
    ] = conjunction_judgement,
) -> Mapping[str, il.LabelProvider[KT, str]]:
    keys = frozenset([lbl.split("_")[0] for lbl in prov.labelset])

    def get_judgement(prov: il.LabelProvider[KT, str]) -> il.LabelProvider[KT, str]:
        tuples = [
            (idx, frozenset([str(judgement_function(labels_to_symbols(lbls)))]))
            for idx, lbls in prov.items()
        ]
        new_prov = il.MemoryLabelProvider.from_tuples(tuples)
        return new_prov

    def get_subset(
        prov: il.LabelProvider[KT, str], key: str
    ) -> il.LabelProvider[KT, str]:
        tuples = [
            (idx, frozenset([lbl for lbl in lbls if key in lbl]))
            for idx, lbls in prov.items()
        ]
        new_prov = il.MemoryLabelProvider.from_tuples(tuples)
        return new_prov

    return {
        **{k: get_subset(prov, k) for k in keys},
        **{"judgement": get_judgement(prov)},
    }

def binarizer(symbols: Mapping[str, Trinary]) -> Mapping[str, bool]:
    return value_map(weakly, symbols)

class SubSetter(abc.ABC, Generic[KT]):
    @abc.abstractmethod
    def __call__(self, prov: il.LabelProvider[KT, str], key: str) -> il.LabelProvider[KT, str]:
        raise NotImplementedError

class DefaultSubSetter(SubSetter[KT], Generic[KT]):
    def __call__(self, prov: il.LabelProvider[KT, str], key: str) -> il.LabelProvider[KT, str]:
        def symbol_yielder():
            for idx, lbls in prov.items():
                symbols = labels_to_symbols(lbls)
                yield (idx, frozenset([str(symbols[key])]))
        tuples = list(symbol_yielder())
        if tuples:
            new_prov = il.MemoryLabelProvider.from_tuples(tuples)
            return new_prov
        return il.MemoryLabelProvider(["True", "False", "Unknown"], dict(), dict())
    

class BinarySubSetter(SubSetter[KT], Generic[KT]):
    def __call__(self, prov: il.LabelProvider[KT, str], key: str) -> il.LabelProvider[KT, str]:
        def symbol_yielder():
            for idx, lbls in prov.items():
                symbols = binarizer(labels_to_symbols(lbls))
                yield (idx, frozenset([str(symbols[key])]))
        tuples = list(symbol_yielder())
        if tuples:
            new_prov = il.MemoryLabelProvider.from_tuples(tuples)
            return new_prov
        return il.MemoryLabelProvider(["True", "False"], dict(), dict())

def get_subset(
        prov: il.LabelProvider[KT, str], key: str
    ) -> il.LabelProvider[KT, str]:
    def symbol_yielder():
        for idx, lbls in prov.items():
            symbols = labels_to_symbols(lbls)
            yield (idx, frozenset([str(symbols[key])]))
    tuples = list(symbol_yielder())
    if tuples:
        new_prov = il.MemoryLabelProvider.from_tuples(tuples)
        return new_prov
    return il.MemoryLabelProvider(["True", "False", "Unknown"], dict(), dict())

def combine_subsets(
       provs: Mapping[str, il.LabelProvider[KT, str]]
    ) -> il.LabelProvider[KT, str]:
    criteria = sorted(list(provs.keys()))
    inss = union(*[frozenset(prov.keys()) for prov in provs.values()])
    def symbol_yielder():
        for idx in inss:            
            symbols = {ck: string_to_symbol(next(iter(provs[ck][idx]))) for ck in criteria}
            labels = symbols_to_labels(symbols)
            yield (idx, labels)
    tuples = list(symbol_yielder())
    if tuples:
        new_prov = il.MemoryLabelProvider.from_tuples(tuples)
        return new_prov
    labelset = frozenset([f"{ck}_{st}" for (ck, st) in product(criteria, ["Y", "N", "U"])])
    return il.MemoryLabelProvider(labelset, dict(), dict())


def check_fit_ability(inss: il.InstanceProvider[IT, KT, Any, Any, Any], labels: il.LabelProvider[KT, LT]) -> bool:
    lengths = all([len(labels.get_instances_by_label(lbl).intersection(inss)) > 0 for lbl in labels.labelset])
    return lengths


class AbstractCriteriaSplitter(abc.ABC, Generic[KT, _T, LT]):
    @abc.abstractmethod
    def __call__(self, prov: il.LabelProvider[KT, _T]) -> Mapping[str, il.LabelProvider[KT, LT]]:
        raise NotImplementedError
    
class SubLabelProviderSplitter(AbstractCriteriaSplitter[KT, str, str]):
    subsetter: SubSetter[KT]
    
    def __init__(self, subsetter: SubSetter[KT] = DefaultSubSetter[KT]()) -> None:
        self.subsetter = subsetter


    def __call__(self, prov: il.LabelProvider[KT, str]) -> Mapping[str, il.LabelProvider[KT, str]]:
        keys = frozenset([lbl.split("_")[0] for lbl in prov.labelset])
        result = {k: self.subsetter(prov, k) for k in keys}
        return result
    
class ProbaProvider(abc.ABC, Generic[KT, LT]):
    labelset: FrozenSet[LT]

    @abc.abstractmethod
    def above(self, label: LT, value: float) -> FrozenSet[KT]:
        raise NotImplementedError
    @abc.abstractmethod
    def below(self, label: LT, value: float) -> FrozenSet[KT]:
        raise NotImplementedError

    @abc.abstractmethod
    def proba(self, key: KT, label: LT) -> float:
        raise NotImplementedError

    def probas(self, keys: Sequence[KT], label) -> Sequence[Tuple[KT, float]]:
        return [(k, self.proba(k, label)) for k in keys]
    
    @abc.abstractmethod
    def quantile(self, label: LT, value: float) -> float:
        raise NotImplementedError
    
    def above_quantile(self, label: LT, value: float) -> FrozenSet[KT]:
        quantile = self.quantile(label, value)
        return self.above(label, quantile)
    
class SimpleProbaProvider(ProbaProvider[KT, LT], Generic[KT, LT]):
    label_dict: Mapping[LT, Sequence[Tuple[KT, float]]]
    key_dict: Mapping[KT, Mapping[LT, float]]
    
    def __init__(self, probas: Sequence[Tuple[KT, FrozenSet[Tuple[LT, float]]]]):
        self.key_dict = {k: dict(v) for (k,v) in probas}
        self.keys = frozenset((k for (k,_) in probas))
        self.labelset = frozenset({k for v in itertools.islice(self.key_dict.values(), 1) for k in v.keys()})
        self.df = pd.DataFrame.from_records(self.key_dict).T

    def above(self, label: LT, value: float) -> FrozenSet[KT]:
        if label in self.labelset:
            df_slice = self.df[self.df[label] >= value]
            return frozenset(df_slice.index)
        return frozenset()
    
    def below(self, label: LT, value: float) -> FrozenSet[KT]:
        if label in self.labelset:
            df_slice = self.df[self.df[label] >= value]
            return frozenset(df_slice.index)
        return frozenset()
    
    def proba(self, key: KT, label: LT) -> float:
        return self.key_dict[key][label]
    
    def quantile(self, label: LT, value: float) -> float:
        return self.df[label].quantile(value)
    
def get_labeldiff(labels_a: FrozenSet[LT], labels_b: FrozenSet[LT]) -> Mapping[str, Tuple[Trinary, Trinary]]:
    symbols_a = labels_to_symbols(labels_a) 
    symbols_b = labels_to_symbols(labels_b)
    ret: Dict[str, Tuple[Trinary, Trinary]] = dict()
    for ck, symbol_a in symbols_a.items():
        symbol_b = symbols_b[ck]
        if symbol_a is not symbol_b:
            ret[ck] = (symbol_a, symbol_b)
    return ret

def num_of_exclusions(labels: FrozenSet[LT]) -> int:
    symbols = labels_to_symbols(labels)
    exclusions = [k for k,v in symbols.items() if v is False]
    return len(exclusions)

def strong_inclusion_change(labeldiff: Mapping[str, Tuple[Trinary, Trinary]]) -> bool:
    ret = False
    for (s_a, s_b) in labeldiff.values():
        if s_b is False:
            return False
        if s_a is False and s_b is True:
            ret = True
    return ret
