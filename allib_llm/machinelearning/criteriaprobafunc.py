from __future__ import annotations

from typing import Any, Callable, FrozenSet, Generic, Mapping, Sequence, Tuple, TypeVar

import numpy as np
import numpy.typing as npt
from instancelib.instances.base import Instance
from instancelib.labels.base import LabelProvider
from instancelib.labels.encoder import LabelEncoder, MultilabelDictionaryEncoder
from instancelib.machinelearning.wrapper import numpy_ova_threshold
from instancelib.typehints import DT, KT, LMT, LT, LVT, PMT, RT, VT
from trinary import Trinary, Unknown, weakly
from typing_extensions import Self

from ..activelearning.wsa import Acceptance
from ..machinelearning.mockclassifier import MockClassifierWrapper
from ..utils.instances import (
    ProbaProvider,
    conjunction_judgement,
    labels_to_symbols,
    string_to_symbol,
)

IT = TypeVar("IT", bound="Instance[Any, Any, Any, Any]")

class MockProbaFunction(Generic[KT, LT]):
    labelset: Sequence[LT]
    labels: LabelProvider[KT, LT]
    judgement_function: Callable[[Mapping[str, Trinary]], Trinary]

    def __init__(
        self,
        labelprovider: LabelProvider[KT, LT],
        pos_label: LT,
        neg_label: LT,
        judgement_function: Callable[
            [Mapping[str, Trinary]], Trinary
        ] = conjunction_judgement,
    ):
        self.labels = labelprovider
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.labelset = (neg_label, pos_label)
        self.judgement_function = judgement_function

    def symbols_to_score(self, symbols: Mapping[str, Trinary]) -> float:
        if weakly(self.judgement_function(symbols)):
            return 0.50 + sum([0.10 for symbol in symbols.values() if symbol is True])
        return 0.50 - sum([0.10 for symbol in symbols.values() if symbol is False])

    def __call__(self, keys: Sequence[KT]) -> npt.NDArray[np.float64]:
        ret = np.zeros((len(keys), 2))
        for i, key in enumerate(keys):
            labels = self.labels[key]
            symbols = labels_to_symbols(labels)
            prob_pos = self.symbols_to_score(symbols)
            ret[i, 0] = 1 - prob_pos
            ret[i, 1] = prob_pos
        return ret

    @property
    def encoder(self) -> LabelEncoder:
        inverted_mapping = {self.pos_label: 1, self.neg_label: 0}
        encoder = MultilabelDictionaryEncoder(inverted_mapping)
        return encoder


def combine(
    llm_symbols: Mapping[str, Trinary],
    acceptance: Mapping[str, Acceptance],
    correction: Mapping[str, Trinary],
) -> Mapping[str, Trinary]:
    return {
        ck: status if acceptance[ck] == Acceptance.ACCEPT else correction[ck]
        for ck, status in llm_symbols.items()
    }


class CorrectedProbaFunction(Generic[KT, LT]):
    labelset: Sequence[LT]
    criteria: Sequence[str]
    llm_labels: LabelProvider[KT, str]
    acceptance_predictions: Mapping[str, ProbaProvider[KT, Acceptance]]
    answer_correction_predictions: Mapping[str, ProbaProvider[KT, str]]
    classifier_predictions: Mapping[str, LabelProvider[KT, str]]
    judgement_function: Callable[[Mapping[str, Trinary]], Trinary]

    def __init__(
        self,
        llm_labels: LabelProvider[KT, str],
        classifier_probas: Mapping[str, ProbaProvider[KT, str]],
        classifier_predictions: Mapping[str, LabelProvider[KT, str]],
        acceptance_predictions: Mapping[str, ProbaProvider[KT, Acceptance]],
        answer_correction_predictions: Mapping[str, ProbaProvider[KT, str]],
        pos_label: LT,
        neg_label: LT,
        judgement_function: Callable[
            [Mapping[str, Trinary]], Trinary
        ] = conjunction_judgement,
    ):
        self.llm_labels = llm_labels
        self.classifier_probas = classifier_probas
        self.classifier_predictions = classifier_predictions
        self.acceptance_predictions = acceptance_predictions
        self.answer_correction_predictions = answer_correction_predictions
        self.criteria = tuple(acceptance_predictions.keys())
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.labelset = (neg_label, pos_label)
        self.judgement_function = judgement_function
        self.accept_threshold = 0.80
        self.uncertain_threshold = 0.60
        self.reject_threshold = 0.30

    # def symbols_to_score(
    #     self, symbols: Mapping[str, Trinary], skip: FrozenSet[str] = frozenset()
    # ) -> float:
    #     filtered = {k: v for k, v in symbols.items() if k not in skip}
    #     if weakly(self.judgement_function(symbols)):
    #         return 
    #     return 0.50 - sum([0.10 for symbol in filtered.values() if symbol is False])
    
    def symbol_to_score_component(self, status: Trinary, accept_proba: float) -> Tuple[Trinary, float]:
        if accept_proba < self.uncertain_threshold:
            if status is False:
                return Unknown, accept_proba
        if accept_proba < self.reject_threshold:
            if status is True:
                return Unknown, accept_proba
            if status is False:
                return Unknown, accept_proba
        return status, accept_proba
    
    def score(self, tup: Tuple[Trinary, float]):
        symbol, proba = tup
        if symbol is True:
            score = (0.75 + 0.25 * proba)
        if symbol is Unknown:
            score = (0.50 + 0.25 * proba)
        if symbol is False:
            score = (0.5 * (1 - proba))  
        return score
    
    
    
    def any_exclude(self, symbols: Mapping[str, Trinary], exclude:FrozenSet[str] = frozenset()) -> bool:
        return any([s is False for k,s in symbols.items() if k not in exclude])


    def __call__(self, keys: Sequence[KT]) -> npt.NDArray[np.float64]:
        ret = np.zeros((len(keys), 2))
        excluded_keys = {ck: clf.get_instances_by_label("False").intersection(keys) for ck,clf in self.classifier_predictions.items()}
        excluded_criteria_from_ranking = frozenset([ck for ck,exkeys in excluded_keys.items() if len(exkeys)/len(keys) >= 0.97])
        for i, key in enumerate(keys):
            labels = self.llm_labels[key]
            symbols = labels_to_symbols(labels)
            acceptance = {
                ck: self.acceptance_predictions[ck].proba(key, Acceptance.ACCEPT)
                for ck in self.criteria
            }
           
            components = [
                self.score((symbols[ck], acceptance[ck]))
                for ck in self.criteria
            ]
            if self.any_exclude(symbols, excluded_criteria_from_ranking):
                prob_pos = np.min(components)
            else:
                prob_pos = np.mean(components)
            # corrected_symbols = combine(symbols, acceptance, correction_labels)
            # prob_pos = self.symbols_to_score(corrected_symbols)
            ret[i, 0] = 1 - prob_pos
            ret[i, 1] = prob_pos
        return ret

    @property
    def encoder(self) -> LabelEncoder:
        inverted_mapping = {self.pos_label: 1, self.neg_label: 0}
        encoder = MultilabelDictionaryEncoder(inverted_mapping)
        return encoder

class ClassifierProbaFunction(Generic[KT, LT]):
    labelset: Sequence[LT]
    criteria: Sequence[str]
    classifier_predictions: Mapping[str, LabelProvider[KT, str]]
    classifier_probas: Mapping[str, ProbaProvider[KT, str]]
    acceptance_predictions: Mapping[str, ProbaProvider[KT, Acceptance]]
    answer_correction_predictions: Mapping[str, ProbaProvider[KT, str]]

    judgement_function: Callable[[Mapping[str, Trinary]], Trinary]

    def __init__(
        self,
        classifier_predictions: Mapping[str, LabelProvider[KT, str]],
        classifier_probas: Mapping[str, ProbaProvider[KT, str]],
        pos_label: LT,
        neg_label: LT,
        judgement_function: Callable[
            [Mapping[str, Trinary]], Trinary
        ] = conjunction_judgement,
    ):
        self.classifier_predictions = classifier_predictions
        self.classifier_probas = classifier_probas
        self.criteria = tuple(classifier_predictions.keys())
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.labelset = (neg_label, pos_label)
        self.judgement_function = judgement_function
        self.accept_threshold = 0.80
        self.uncertain_threshold = 0.60
        self.reject_threshold = 0.30

    def score(self, tup: Tuple[Trinary, float]):
        symbol, proba = tup
        if symbol is True:
            score = (0.50 + 0.50 * proba)
        if symbol is Unknown:
            score = (0.50 + 0.10 * proba)
        if symbol is False:
            if proba <= 0.50:
                score = 0.50 + 0.10 * (1-proba)
            else:
                score = (0.5 * (1 - proba))  
        return score
    
    def any_exclude(self, symbols: Mapping[str, Trinary], exclude:FrozenSet[str] = frozenset()) -> bool:
        return any([s is False for k,s in symbols.items() if k not in exclude])

    def __call__(self, keys: Sequence[KT]) -> npt.NDArray[np.float64]:
        ret = np.zeros((len(keys), 2))
        excluded_keys = {ck: clf.get_instances_by_label("False").intersection(keys) for ck,clf in self.classifier_predictions.items()}
        excluded_criteria_from_ranking = frozenset([ck for ck,exkeys in excluded_keys.items() if len(exkeys)/len(keys) >= 0.97])
        for i, key in enumerate(keys):
            symbols = { ck: string_to_symbol(pred) 
                 for ck, preds in self.classifier_predictions.items() for pred in preds[key]
            }
            components = [
                self.score((symbols[ck], self.classifier_probas[ck].proba(key, str(symbols[ck]))))
                for ck in self.criteria
            ]
            if self.any_exclude(symbols, exclude=excluded_criteria_from_ranking):
                prob_pos = np.min(components)
            else:
                prob_pos = np.mean(components)
            # corrected_symbols = combine(symbols, acceptance, correction_labels)
            # prob_pos = self.symbols_to_score(corrected_symbols)
            ret[i, 0] = 1 - prob_pos
            ret[i, 1] = prob_pos
        return ret

    @property
    def encoder(self) -> LabelEncoder:
        inverted_mapping = {self.pos_label: 1, self.neg_label: 0}
        encoder = MultilabelDictionaryEncoder(inverted_mapping)
        return encoder


class MockProbaClassifier(
    MockClassifierWrapper[IT, KT, DT, VT, RT, LT, LVT, LMT, PMT],
    Generic[IT, KT, DT, VT, RT, LT, LVT, LMT, PMT],
):

    @classmethod
    def from_provider(
        cls,
        provider: LabelProvider[KT, LT],
        pos_label: LT,
        neg_label: LT,
        judgement_function: Callable[
            [Mapping[str, Trinary]], Trinary
        ] = conjunction_judgement,
    ) -> Self:
        mock_model = MockProbaFunction(
            provider, pos_label, neg_label, judgement_function
        )
        threshold = numpy_ova_threshold(0.5)
        wrapped = cls(mock_model, threshold, mock_model.encoder)  # type: ignore
        return wrapped

    @classmethod
    def from_corrections(
        cls,
        llm_labels: LabelProvider[KT, str],
        classifier_predictions: Mapping[str, LabelProvider[KT, str]],
        classifier_probas: Mapping[str, ProbaProvider[KT, str]],
        acceptance_predictions: Mapping[str, ProbaProvider[KT, Acceptance]],
        answer_correction_predictions: Mapping[str, ProbaProvider[KT, str]],
        pos_label: LT,
        neg_label: LT,
        judgement_function: Callable[
            [Mapping[str, Trinary]], Trinary
        ] = conjunction_judgement,
    ) -> Self:
        mock_model = CorrectedProbaFunction(
            llm_labels,
            classifier_probas,
            classifier_predictions,
            acceptance_predictions,
            answer_correction_predictions,
            pos_label,
            neg_label,
            judgement_function,
        )
        threshold = numpy_ova_threshold(0.5)
        wrapped = cls(mock_model, threshold, mock_model.encoder)  # type: ignore
        return wrapped
    
    @classmethod
    def from_classifiers(cls, 
                         classifier_predictions: Mapping[str, LabelProvider[KT, str]],
                         classifier_probas: Mapping[str, ProbaProvider[KT, str]],
                         pos_label: LT,
                         neg_label: LT,
                            judgement_function: Callable[
                                [Mapping[str, Trinary]], Trinary
                            ] = conjunction_judgement) -> Self:
        mock_model = ClassifierProbaFunction(
            classifier_predictions,
            classifier_probas,
            pos_label,
            neg_label,
            judgement_function,
        )
        threshold = numpy_ova_threshold(0.5)
        wrapped = cls(mock_model, threshold, mock_model.encoder)  # type: ignore
        return wrapped