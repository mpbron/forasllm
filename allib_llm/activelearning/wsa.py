import collections
import random
from enum import Enum
from math import ceil
from typing import (
    Any,
    Callable,
    Deque,
    Generic,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import instancelib as il
import numpy.typing as npt
from allib.activelearning.autotar import PSEUDO_INS_PROVIDER, AutoTarLearner
from allib.activelearning.poolbased import PoolBasedAL
from allib.environment.base import AbstractEnvironment
from allib.typehints.typevars import DT, IT, KT, LT, RT, VT
from instancelib.typehints.typevars import KT, LT
from instancelib.utils.func import list_unzip, sort_on
from numpy._typing import NDArray
from typing_extensions import Self

from ..analysis.stats import CriteriaStatisticsSlim
from ..utils.instances import LabelTransformer


class Acceptance(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"


_T = TypeVar("_T")


def pseudo_from_metadata(
    env: AbstractEnvironment[IT, KT, DT, VT, RT, LT]
) -> AbstractEnvironment[IT, KT, DT, VT, RT, LT]:
    if "inclusions" in env.metadata:
        criteria = [
            question
            for critdict in env.metadata["inclusions"]
            for question in critdict.values()
        ]
        pseudo_article = {"title": "", "abstract": " ".join(criteria)}
        ins = env.create(data=pseudo_article, vector=None)
        env.create_named_provider(PSEUDO_INS_PROVIDER, [ins.identifier])
    return env


def get_agreement_labels(
    labeled: il.InstanceProvider[Any, KT, Any, Any, Any],
    user_labels: il.LabelProvider[KT, LT],
    wsa_labels: il.LabelProvider[KT, LT],
) -> il.LabelProvider[KT, Acceptance]:
    agreement_provider: il.MemoryLabelProvider[KT, Acceptance] = il.MemoryLabelProvider(
        (Acceptance.ACCEPT, Acceptance.REJECT), dict()
    )
    for ins in labeled:
        user_label = user_labels[ins]
        wsa_label = wsa_labels[ins]
        if user_label == wsa_label:
            agreement_provider.set_labels(ins, Acceptance.ACCEPT)
        else:
            agreement_provider.set_labels(ins, Acceptance.REJECT)
    return agreement_provider


def combine_right_override(
    a: il.LabelProvider[KT, LT], b: il.LabelProvider[KT, LT]
) -> il.LabelProvider[KT, LT]:
    new_provider = il.MemoryLabelProvider.from_provider(b)
    for k, lbls in a.items():
        if k not in new_provider:
            new_provider.set_labels(k, *lbls)
    return new_provider


def labelprovidermap(
    func: Callable[[LT], _T], prov: il.LabelProvider[KT, LT]
) -> il.LabelProvider[KT, _T]:
    tuples = [(k, frozenset([func(lbl) for lbl in lbls])) for k, lbls in prov.items()]
    new_provider = il.MemoryLabelProvider[KT, _T].from_tuples(tuples)
    return new_provider


class AutoTARReduced(
    AutoTarLearner[IT, KT, DT, VT, RT, LT], Generic[IT, KT, DT, VT, RT, LT]
):
    labeltransformer: LabelTransformer[KT, LT]

    def __init__(
        self,
        env: AbstractEnvironment[IT, KT, DT, VT, RT, LT],
        classifier: il.AbstractClassifier[
            IT, KT, DT, VT, RT, LT, npt.NDArray[Any], npt.NDArray[Any]
        ],
        labeltransformer: LabelTransformer[KT, LT],
        pos_label: LT,
        neg_label: LT,
        k_sample: int,
        batch_size: int,
        *_,
        seed: int = 0,
        chunk_size: int = 2000,
        identifier: str | None = None,
        **__,
    ) -> None:
        super().__init__(
            env,
            classifier,
            pos_label,
            neg_label,
            k_sample,
            batch_size,
            *_,
            seed=seed,
            chunk_size=chunk_size,
            identifier=identifier,
            **__,
        )
        self.labeltransformer = labeltransformer
        self._stats = CriteriaStatisticsSlim(
            self.labeltransformer, self.pos_label, self.neg_label
        )

    def _temp_augment_and_train(self):
        temp_labels = self.labeltransformer(
            self.env.labels, self.pos_label, self.neg_label
        )
        sampled_non_relevant = self._provider_sample(self.env.unlabeled)
        if PSEUDO_INS_PROVIDER in self.env:
            pseudo_docs = self.env[PSEUDO_INS_PROVIDER]
            for ins_key in pseudo_docs:
                temp_labels.set_labels(ins_key, self.pos_label)
        else:
            pseudo_docs = self.env.create_bucket([])
        for ins_key in sampled_non_relevant:
            temp_labels.set_labels(ins_key, self.neg_label)
        train_set = self.env.combine(
            sampled_non_relevant, self.env.labeled, pseudo_docs
        )
        self.classifier.fit_provider(train_set, temp_labels)

    @classmethod
    def builder(
        cls,
        classifier_builder: Callable[
            [AbstractEnvironment[IT, KT, DT, VT, RT, LT]],
            il.AbstractClassifier[
                IT, KT, DT, VT, RT, LT, npt.NDArray[Any], npt.NDArray[Any]
            ],
        ],
        label_transformer: LabelTransformer,
        k_sample: int,
        batch_size: int,
        chunk_size: int = 2000,
        initializer: Callable[
            [AbstractEnvironment[IT, KT, DT, VT, RT, LT]],
            AbstractEnvironment[IT, KT, DT, VT, RT, LT],
        ] = pseudo_from_metadata,
        identifier: Optional[str] = None,
    ) -> Callable[..., Self]:
        def builder_func(
            env: AbstractEnvironment[IT, KT, DT, VT, RT, LT],
            pos_label: LT,
            neg_label: LT,
            *_,
            identifier: Optional[str] = identifier,
            **__,
        ):
            env = initializer(env)
            classifier = classifier_builder(env)
            return cls(
                env,
                classifier,
                label_transformer,
                pos_label,
                neg_label,
                k_sample,
                batch_size,
                chunk_size=chunk_size,
                identifier=identifier,
            )

        return builder_func


class AutoTarWSA(
    AutoTarLearner[IT, KT, DT, VT, RT, LT],
    Generic[IT, KT, DT, VT, RT, LT],
):
    _name = "AutoTarWSA"
    wsa_classifier: il.AbstractClassifier[
        IT, KT, DT, VT, RT, LT, npt.NDArray[Any], npt.NDArray[Any]
    ]
    agreement_classifier: il.AbstractClassifier[
        IT, KT, DT, VT, RT, Acceptance, npt.NDArray[Any], npt.NDArray[Any]
    ]
    wsa_predictions: il.LabelProvider[KT, LT]
    acceptance_labels: il.LabelProvider[KT, Acceptance]
    pseudo_acceptance_labels: il.LabelProvider[KT, Acceptance]
    pseudo_labels: il.LabelProvider[KT, LT]
    normal: bool
    labeltransformer: Callable[
        [il.LabelProvider[KT, Any], LT, LT], il.LabelProvider[KT, LT]
    ]

    def __init__(
        self,
        env: AbstractEnvironment[IT, KT, DT, VT, RT, LT],
        classifier: il.AbstractClassifier[
            IT, KT, DT, VT, RT, LT, npt.NDArray[Any], npt.NDArray[Any]
        ],
        trained_classifier: il.AbstractClassifier[
            IT, KT, DT, VT, RT, LT, npt.NDArray[Any], npt.NDArray[Any]
        ],
        agreement_classifier: il.AbstractClassifier[
            IT, KT, DT, VT, RT, Acceptance, npt.NDArray[Any], npt.NDArray[Any]
        ],
        labeltransformer: LabelTransformer[KT, LT],
        pos_label: LT,
        neg_label: LT,
        k_sample: int,
        batch_size: int,
        *_,
        seed: int = 0,
        chunk_size: int = 2000,
        identifier: str | None = None,
        **__,
    ) -> None:
        super().__init__(
            env,
            classifier,
            pos_label,
            neg_label,
            k_sample,
            batch_size,
            *_,
            seed=seed,
            chunk_size=chunk_size,
            identifier=identifier,
            **__,
        )
        self.wsa_classifier = trained_classifier
        self.acceptance_classifier = agreement_classifier
        predictions = self.wsa_classifier.predict(self.env.dataset)
        self.labeltransformer = labeltransformer
        self.wsa_predictions = il.MemoryLabelProvider.from_tuples(predictions)
        self.pseudo_labeled = self.env.create_empty_provider()
        binary_labels = self.labeltransformer(
            self.env.labels, self.pos_label, self.neg_label
        )
        self.binary_wsa_labels = self.labeltransformer(
            self.wsa_predictions, self.pos_label, self.neg_label
        )
        self.pseudo_labels = il.MemoryLabelProvider.from_provider(
            self.env.labels, subset=[]
        )
        self.acceptance_labels = get_agreement_labels(
            self.env.labeled, binary_labels, self.binary_wsa_labels
        )
        self.pseudo_acceptance_labels = il.MemoryLabelProvider(
            (Acceptance.ACCEPT, Acceptance.REJECT), dict()
        )
        self.acceptance_predictions = il.MemoryLabelProvider(
            (Acceptance.ACCEPT, Acceptance.REJECT), dict()
        )
        self.normal = False
        self._stats = CriteriaStatisticsSlim(
            self.labeltransformer, self.pos_label, self.neg_label
        )
        self.last_ret_it = self.it

    def set_as_labeled(self, instance: il.Instance[KT, DT, VT, RT]) -> None:
        super().set_as_labeled(instance)
        self.pseudo_labeled.discard(instance)
        self.pseudo_labels = il.MemoryLabelProvider.from_provider(
            self.pseudo_labels, self.pseudo_labeled.key_list
        )
        binary_labels = self.labeltransformer(
            self.env.labels, self.pos_label, self.neg_label
        )
        binary_wsa_labels = self.labeltransformer(
            self.wsa_predictions, self.pos_label, self.neg_label
        )
        self.acceptance_labels = get_agreement_labels(
            self.env.labeled, binary_labels, binary_wsa_labels
        )

    def get_candidate_set(self) -> il.InstanceProvider[IT, KT, DT, VT, RT]:
        return self.env.create_bucket(
            self.wsa_predictions.get_instances_by_label(self.pos_label).intersection(
                self.env.unlabeled
            )
        )

    def _provider_sample(
        self, provider: il.InstanceProvider[IT, KT, DT, VT, RT]
    ) -> il.InstanceProvider[IT, KT, DT, VT, RT]:
        k_sample = min(self.k_sample, len(provider))
        sampled_keys: Sequence[KT] = self.rng.choice(
            provider.key_list, size=k_sample, replace=False  # type: ignore
        ).tolist()
        sampled_provider = self.env.create_bucket(sampled_keys)
        return sampled_provider

    def _temp_augment_and_train(self):
        binary_labels = self.labeltransformer(
            self.env.labels, self.pos_label, self.neg_label
        )
        binary_pseudo_labels = self.labeltransformer(
            self.pseudo_labels, self.pos_label, self.neg_label
        )
        temp_labels = combine_right_override(binary_pseudo_labels, binary_labels)
        # temp_acc_labels = combine_right_override(self.pseudo_acceptance_labels, self.acceptance_labels)
        train_set = self.env.combine(self.env.labeled, self.pseudo_labeled)
        self.classifier.fit_provider(train_set, temp_labels)
        # Determine what works
        self.acceptance_classifier.fit_provider(
            self.env.labeled, self.acceptance_labels
        )
        # self.acceptance_classifier.fit_provider(train_set, temp_acc_labels)

    def _rank(
        self, provider: il.InstanceProvider[IT, KT, DT, VT, RT]
    ) -> Sequence[Tuple[KT, float]]:
        if self.it >= 1:
            previous_round: Sequence[KT] = [
                k
                for it in range(max(0,self.last_ret_it), max(0,self.it-1))
                for k in self.sampled_sets[it]
            ]
        else:
            previous_round: Sequence[KT] = tuple()
        prvkeys = frozenset(provider)
        exclude_previous_round = prvkeys.difference(previous_round)
        new_provider = self.env.create_bucket(exclude_previous_round)
        return super()._rank(new_provider)

    def update_sample(self) -> Deque[KT]:
        if not self.current_sample:
            self.stats.update(self)
            self._temp_augment_and_train()
            ranking = self._rank(self.env.unlabeled)
            sample = self._sample(ranking)

            # Store current sample
            self.current_sample = collections.deque(sample)

            # Determine Acceptance Predictions for sample
            sample_provider = self.env.create_bucket(self.current_sample)
            acc_preds = self.acceptance_classifier.predict(sample_provider)
            self.acceptance_predictions = il.MemoryLabelProvider.from_tuples(acc_preds)

            # Record keeping
            self.rank_history[self.it] = self._to_history(ranking)
            self.sampled_sets[self.it] = tuple(sample)
            self.batch_sizes[self.it] = self.batch_size

            # Increment batch_size for next for train iteration
            # self.batch_size += ceil(self.batch_size / 10)
            self.it += 1
        return self.current_sample

    def _pseudo_label(self, ins: Union[KT, il.Instance[KT, DT, VT, RT]]) -> None:
        wsa_labels = self.wsa_predictions[ins]
        self.pseudo_labels.set_labels(ins, *wsa_labels)
        self.pseudo_acceptance_labels.set_labels(ins, Acceptance.ACCEPT)

    def _pseudo_neg_label(self, ins: Union[KT, il.Instance[KT, DT, VT, RT]]) -> None:
        wsa_neg_example = next(
            iter(self.binary_wsa_labels.get_instances_by_label(self.neg_label))
        )
        wsa_ex_labels = self.pseudo_labels[wsa_neg_example]
        self.pseudo_labels.set_labels(ins, *wsa_ex_labels)
        self.pseudo_acceptance_labels.set_labels(ins, Acceptance.REJECT)

    def __next__(self) -> IT:
        if self.env.unlabeled.empty:
            raise StopIteration
        self.update_sample()
        while self.current_sample:
            ins_key = self.current_sample.popleft()
            if ins_key not in self.env.labeled:
                ins = self.env.dataset[ins_key]
                probas = self.acceptance_classifier.predict_proba([ins])
                proba_dict = dict(probas[0][1])
                chance = 0.70
                if proba_dict[Acceptance.ACCEPT] >= chance:
                    if self.binary_wsa_labels[ins] == self.pos_label:
                        return ins
                    self._pseudo_label(ins)
                    self.pseudo_labeled.add(ins)
                elif (
                    1.0 - proba_dict[Acceptance.ACCEPT] >= chance
                    and self.binary_wsa_labels[ins] == self.pos_label
                ):
                    self._pseudo_neg_label(ins)
                    self.pseudo_labeled.add(ins)
                else:
                    self.last_ret_it = self.it
                    return ins
        if not self.env.unlabeled.empty:
            return self.__next__()
        raise StopIteration

    @classmethod
    def builder(
        cls,
        classifier_builder: Callable[
            [AbstractEnvironment[IT, KT, DT, VT, RT, LT]],
            il.AbstractClassifier[
                IT, KT, DT, VT, RT, LT, npt.NDArray[Any], npt.NDArray[Any]
            ],
        ],
        trained_classifier_builder: Callable[
            [AbstractEnvironment[IT, KT, DT, VT, RT, LT]],
            il.AbstractClassifier[
                IT, KT, DT, VT, RT, LT, npt.NDArray[Any], npt.NDArray[Any]
            ],
        ],
        acceptance_builder: Callable[
            [AbstractEnvironment[IT, KT, DT, VT, RT, LT]],
            il.AbstractClassifier[
                IT, KT, DT, VT, RT, Acceptance, npt.NDArray[Any], npt.NDArray[Any]
            ],
        ],
        label_transformer: LabelTransformer,
        k_sample: int,
        batch_size: int,
        chunk_size: int = 2000,
        initializer: Callable[
            [AbstractEnvironment[IT, KT, DT, VT, RT, LT]],
            AbstractEnvironment[IT, KT, DT, VT, RT, LT],
        ] = pseudo_from_metadata,
        identifier: Optional[str] = None,
    ) -> Callable[..., Self]:
        def builder_func(
            env: AbstractEnvironment[IT, KT, DT, VT, RT, LT],
            pos_label: LT,
            neg_label: LT,
            *_,
            identifier: Optional[str] = identifier,
            **__,
        ):
            env = initializer(env)
            classifier = classifier_builder(env)
            trained_classifier = trained_classifier_builder(env)
            acceptance_classifier = acceptance_builder(env)
            return cls(
                env,
                classifier,
                trained_classifier,
                acceptance_classifier,
                label_transformer,
                pos_label,
                neg_label,
                k_sample,
                batch_size,
                chunk_size=chunk_size,
                identifier=identifier,
            )

        return builder_func

    @classmethod
    def builder_alt(
        cls,
        classifier_builder: Callable[
            [AbstractEnvironment[IT, KT, DT, VT, RT, LT]],
            il.AbstractClassifier[
                IT, KT, DT, VT, RT, LT, npt.NDArray[Any], npt.NDArray[Any]
            ],
        ],
        trained_classifier: il.AbstractClassifier[
            IT, KT, DT, VT, RT, LT, npt.NDArray[Any], npt.NDArray[Any]
        ],
        acceptance_builder: Callable[
            [AbstractEnvironment[IT, KT, DT, VT, RT, LT]],
            il.AbstractClassifier[
                IT, KT, DT, VT, RT, Acceptance, npt.NDArray[Any], npt.NDArray[Any]
            ],
        ],
        label_transformer: LabelTransformer[KT, LT],
        k_sample: int,
        batch_size: int,
        chunk_size: int = 2000,
        initializer: Callable[
            [AbstractEnvironment[IT, KT, DT, VT, RT, LT]],
            AbstractEnvironment[IT, KT, DT, VT, RT, LT],
        ] = pseudo_from_metadata,
        identifier: Optional[str] = None,
    ) -> Callable[..., Self]:
        def builder_func(
            env: AbstractEnvironment[IT, KT, DT, VT, RT, LT],
            pos_label: LT,
            neg_label: LT,
            *_,
            identifier: Optional[str] = identifier,
            **__,
        ):
            env = initializer(env)
            classifier = classifier_builder(env)
            acceptance_classifier = acceptance_builder(env)
            return cls(
                env,
                classifier,
                trained_classifier,
                acceptance_classifier,
                label_transformer,
                pos_label,
                neg_label,
                k_sample,
                batch_size,
                chunk_size=chunk_size,
                identifier=identifier,
            )

        return builder_func


class LLMPreferred(
    AutoTARReduced[IT, KT, DT, VT, RT, LT], Generic[IT, KT, DT, VT, RT, LT]
):
    _name = "LLMPreferred"
    
    def __init__(
        self,
        env: AbstractEnvironment[IT, KT, DT, VT, RT, LT],
        classifier: il.AbstractClassifier[
            IT, KT, DT, VT, RT, LT, NDArray[Any], NDArray[Any]
        ],
        trained_classifier: il.AbstractClassifier[
            IT, KT, DT, VT, RT, LT, NDArray[Any], NDArray[Any]
        ],
        labeltransformer: LabelTransformer[KT, LT],
        pos_label: LT,
        neg_label: LT,
        k_sample: int,
        batch_size: int,
        *_,
        seed: int = 0,
        chunk_size: int = 2000,
        identifier: str | None = None,
        **__,
    ) -> None:
        super().__init__(
            env,
            classifier,
            labeltransformer,
            pos_label,
            neg_label,
            k_sample,
            batch_size,
            *_,
            seed=seed,
            chunk_size=chunk_size,
            identifier=identifier,
            **__,
        )
        self.trained_classifier = trained_classifier
        predictions = self.trained_classifier.predict(self.env.dataset)
        self.predictions = il.MemoryLabelProvider.from_tuples(predictions)
        self.normal = False

    def set_as_labeled(self, instance: il.Instance[KT, DT, VT, RT]) -> None:
        self.predictions.remove_labels(instance, *self.env.labels.labelset)
        self.predictions.set_labels(instance, *self.env.labels[instance.identifier])
        return super().set_as_labeled(instance)

    def get_candidate_set(self) -> il.InstanceProvider[IT, KT, DT, VT, RT]:
        binary_predictions = self.labeltransformer(self.predictions, self.pos_label, self.neg_label)
        return self.env.create_bucket(
            binary_predictions.get_instances_by_label(self.pos_label).intersection(
                self.env.unlabeled
            )
        )

    def _provider_sample(
        self, provider: il.InstanceProvider[IT, KT, DT, VT, RT]
    ) -> il.InstanceProvider[IT, KT, DT, VT, RT]:
        k_sample = min(self.k_sample, len(provider))
        sampled_keys: Sequence[KT] = self.rng.choice(
            provider.key_list, size=k_sample, replace=False  # type: ignore
        ).tolist()
        sampled_provider = self.env.create_bucket(sampled_keys)
        return sampled_provider

    def _temp_augment_and_train(self):
        binary_labels = self.labeltransformer(self.env.labels, self.pos_label, self.neg_label)
        binary_predictions = self.labeltransformer(self.predictions, self.pos_label, self.neg_label)
        llm_irrelevant = self.env.create_bucket(
            binary_predictions.get_instances_by_label(self.neg_label)
        )
        sampled_non_relevant = self._provider_sample(llm_irrelevant)
        if PSEUDO_INS_PROVIDER in self.env:
            pseudo_docs = self.env[PSEUDO_INS_PROVIDER]
            for ins_key in pseudo_docs:
                binary_labels.set_labels(ins_key, self.pos_label)
        else:
            pseudo_docs = self.env.create_bucket([])
        for ins_key in sampled_non_relevant:
            binary_labels.set_labels(ins_key, self.neg_label)
        train_set = self.env.combine(
            sampled_non_relevant, self.env.labeled, pseudo_docs
        )
        self.classifier.fit_provider(train_set, binary_labels)

    def update_sample(self) -> Deque[KT]:
        if not self.current_sample:
            self.stats.update(self)
            self._temp_augment_and_train()
            candidates = self.get_candidate_set()
            if candidates:
                ranking = self._rank(candidates)
            else:
                ranking = self._rank(self.env.unlabeled)
            sample = self._sample(ranking)

            # Store current sample
            self.current_sample = collections.deque(sample)

            # Record keeping
            self.rank_history[self.it] = self._to_history(ranking)
            self.sampled_sets[self.it] = tuple(sample)
            self.batch_sizes[self.it] = self.batch_size

            # Increment batch_size for next for train iteration
            self.batch_size += ceil(self.batch_size / 10)
            self.it += 1
        return self.current_sample

    @classmethod
    def builder(
        cls,
        classifier_builder: Callable[
            [AbstractEnvironment[IT, KT, DT, VT, RT, LT]],
            il.AbstractClassifier[
                IT, KT, DT, VT, RT, LT, npt.NDArray[Any], npt.NDArray[Any]
            ],
        ],
        trained_classifier_builder: Callable[
            [AbstractEnvironment[IT, KT, DT, VT, RT, LT]],
            il.AbstractClassifier[
                IT, KT, DT, VT, RT, LT, npt.NDArray[Any], npt.NDArray[Any]
            ],
        ],
        label_transformer: LabelTransformer,
        k_sample: int,
        batch_size: int,
        chunk_size: int = 2000,
        initializer: Callable[
            [AbstractEnvironment[IT, KT, DT, VT, RT, LT]],
            AbstractEnvironment[IT, KT, DT, VT, RT, LT],
        ] = pseudo_from_metadata,
        identifier: Optional[str] = None,
    ) -> Callable[..., Self]:
        def builder_func(
            env: AbstractEnvironment[IT, KT, DT, VT, RT, LT],
            pos_label: LT,
            neg_label: LT,
            *_,
            identifier: Optional[str] = identifier,
            **__,
        ):
            env = initializer(env)
            classifier = classifier_builder(env)
            trained_classifier = trained_classifier_builder(env)
            return cls(
                env,
                classifier,
                trained_classifier,
                label_transformer,
                pos_label,
                neg_label,
                k_sample,
                batch_size,
                chunk_size=chunk_size,
                identifier=identifier,
            )

        return builder_func

    @classmethod
    def builder_alt(
        cls,
        classifier_builder: Callable[
            [AbstractEnvironment[IT, KT, DT, VT, RT, LT]],
            il.AbstractClassifier[
                IT, KT, DT, VT, RT, LT, npt.NDArray[Any], npt.NDArray[Any]
            ],
        ],
        trained_classifier: il.AbstractClassifier[
            IT, KT, DT, VT, RT, LT, npt.NDArray[Any], npt.NDArray[Any]
        ],
        label_transformer: LabelTransformer[KT, LT],
        k_sample: int,
        batch_size: int,
        chunk_size: int = 2000,
        initializer: Callable[
            [AbstractEnvironment[IT, KT, DT, VT, RT, LT]],
            AbstractEnvironment[IT, KT, DT, VT, RT, LT],
        ] = pseudo_from_metadata,
        identifier: Optional[str] = None,
    ) -> Callable[..., Self]:
        def builder_func(
            env: AbstractEnvironment[IT, KT, DT, VT, RT, LT],
            pos_label: LT,
            neg_label: LT,
            *_,
            identifier: Optional[str] = identifier,
            **__,
        ):
            env = initializer(env)
            classifier = classifier_builder(env)
            return cls(
                env,
                classifier,
                trained_classifier,
                label_transformer,
                pos_label,
                neg_label,
                k_sample,
                batch_size,
                chunk_size=chunk_size,
                identifier=identifier,
            )

        return builder_func