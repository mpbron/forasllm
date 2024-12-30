import abc
import collections
import random
from enum import Enum
from math import ceil
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    FrozenSet,
    Generic,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import instancelib as il
import numpy as np
import numpy.typing as npt
from allib.activelearning.autotar import PSEUDO_INS_PROVIDER, AutoTarLearner
from allib.activelearning.poolbased import PoolBasedAL
from allib.analysis.base import AbstractStatistics, AnnotationStatisticsSlim, StatsMixin
from allib.environment.base import AbstractEnvironment
from allib.typehints.typevars import DT, IT, KT, LT, RT, VT
from allib.utils.numpy import raw_proba_chainer
from instancelib.typehints.typevars import KT, LT
from instancelib.utils.func import list_unzip, sort_on, value_map
from instancelib.utils.to_key import to_key
from trinary import weakly
from typing_extensions import Self

from allib_llm.machinelearning.criteriaprobafunc import MockProbaClassifier

from ..analysis.stats import CriteriaStatisticsSlim, CriteriaStatisticsSlimV2
from ..machinelearning.criteria_llm import CriteriaLLMClassifier
from ..utils.instances import (
    AbstractCriteriaSplitter,
    AbstractLabelTransformer,
    SimpleProbaProvider,
    SubLabelProviderSplitter,
    SubSetter,
    check_fit_ability,
    get_subset,
    labels_to_symbols,
)
from .wsa import (
    Acceptance,
    combine_right_override,
    get_agreement_labels,
    labelprovidermap,
)

_T = TypeVar("_T")


def proba_sequence_to_mapping(
    sequence: Sequence[Tuple[KT, FrozenSet[Tuple[LT, float]]]]
) -> Mapping[KT, Mapping[LT, float]]:
    result = {ins_key: dict(preds) for (ins_key, preds) in sequence}
    return result


class CriteriaTarLearner(
    PoolBasedAL[IT, KT, DT, VT, RT, str],
    StatsMixin[KT, LT],
    abc.ABC,
    Generic[IT, KT, DT, VT, RT, LT],
):
    rank_history: Dict[int, Mapping[KT, int]]
    sampled_sets: Dict[int, Sequence[KT]]
    batch_sizes: Dict[int, int]
    labeltransformer: AbstractLabelTransformer[KT, Any, LT]
    criteria: Sequence[str]
    criteriatransformer: AbstractCriteriaSplitter[KT, Any, str]
    current_sample: Deque[KT]
    classifiers: Mapping[
        str,
        il.AbstractClassifier[
            IT, KT, DT, VT, RT, str, npt.NDArray[Any], npt.NDArray[Any]
        ],
    ]

    rng: np.random.Generator
    _stats: AbstractStatistics[KT, LT]

    _name = "CriteriaTarLearner"

    def __init__(
        self,
        env: AbstractEnvironment[IT, KT, DT, VT, RT, str],
        classifiers: Mapping[
            str,
            il.AbstractClassifier[
                IT, KT, DT, VT, RT, str, npt.NDArray[Any], npt.NDArray[Any]
            ],
        ],
        labeltransformer: AbstractLabelTransformer[KT, Any, LT],
        criteriatransformer: AbstractCriteriaSplitter[KT, Any, str],
        pos_label: LT,
        neg_label: LT,
        batch_size: int,
        *_,
        seed: int = 0,
        chunk_size: int = 2000,
        identifier: Optional[str] = None,
        **__,
    ) -> None:
        super().__init__(env, *_, identifier=identifier, **__)
        # Problem definition
        self.criteria = tuple(sorted(list(classifiers.keys())))
        self.classifiers = classifiers
        self.pos_label = pos_label
        self.neg_label = neg_label

        # For translation between Criteria to Binary Relevance
        self.labeltransformer = labeltransformer
        # For translation between all criteria to Providers with separate tasks
        self.criteriatransformer = criteriatransformer
        # Batch and sample sizes
        self.chunk_size = chunk_size
        self.batch_sizes = dict()
        self.batch_size = batch_size

        # Record keeping for the current sample
        self.it = 0
        self.current_sample = collections.deque()
        self.batch_sizes[self.it] = batch_size

        # Record keeping for recall analysis
        self.rank_history = dict()
        self.sampled_sets = dict()

        # Random generator for sampling
        self.rng = np.random.default_rng(seed)

        # Statistics Logger for Stopping
        self._stats = CriteriaStatisticsSlimV2(self.labeltransformer)

    @property
    def stats(self) -> AbstractStatistics[KT, LT]:
        return self._stats

    @property
    def name(self) -> Tuple[str, Optional[LT]]:
        if self.identifier is not None:
            return self.identifier, self.pos_label
        return self._name, self.pos_label

    def update_ordering(self) -> bool:
        return True

    def _fit(
        self,
        provider: il.InstanceProvider[IT, KT, DT, VT, RT],
        labels: il.LabelProvider[KT, str],
    ) -> None:
        sublabelproviders = self.criteriatransformer(labels)
        for key, sublabels in sublabelproviders.items():
            clf = self.classifiers[key]
            clf.fit_provider(provider, sublabels)

    def _predict(
        self, provider: il.InstanceProvider[IT, KT, DT, VT, RT]
    ) -> Mapping[str, Mapping[KT, Mapping[str, float]]]:
        def clf_pred(
            clf: il.AbstractClassifier[
                IT, KT, DT, VT, RT, str, npt.NDArray[Any], npt.NDArray[Any]
            ]
        ) -> Mapping[KT, Mapping[str, float]]:
            probas = clf.predict_proba(provider)
            return proba_sequence_to_mapping(probas)

        return value_map(clf_pred, self.classifiers)

    def _predict_stats(
        self, provider: il.InstanceProvider[IT, KT, DT, VT, RT]
    ) -> Mapping[str, il.LabelProvider[KT, str]]:
        def clf_pred(
            clf: il.AbstractClassifier[
                IT, KT, DT, VT, RT, str, npt.NDArray[Any], npt.NDArray[Any]
            ]
        ) -> il.LabelProvider[KT, str]:
            preds = clf.predict(provider)
            return il.MemoryLabelProvider.from_tuples(preds)

        return value_map(clf_pred, self.classifiers)

    @abc.abstractmethod
    def _rank(
        self, provider: il.InstanceProvider[IT, KT, DT, VT, RT]
    ) -> Sequence[Tuple[KT, float]]:
        raise NotImplementedError

    def _sample(self, distribution: Sequence[Tuple[KT, float]]) -> Sequence[KT]:
        keys, _ = list_unzip(distribution)
        sample = keys[0 : self.batch_size]
        return sample

    @classmethod
    def _to_history(cls, ranking: Sequence[Tuple[KT, float]]) -> Mapping[KT, int]:
        keys, _ = list_unzip(ranking)
        history = {k: i for i, k in enumerate(keys, start=1)}
        return history

    def update_sample(self) -> Deque[KT]:
        if not self.current_sample:
            self.stats.update(self)  # type: ignore
            self._fit(self.env.labeled, self.env.labels)
            ranking = self._rank(self.env.unlabeled)
            sample = self._sample(ranking)

            # Store current sample
            self.current_sample = collections.deque(sample)

            # Record keeping
            self.rank_history[self.it] = self._to_history(ranking)
            self.sampled_sets[self.it] = tuple(sample)
            self.batch_sizes[self.it] = self.batch_size

            self.it += 1
        return self.current_sample

    def __next__(self) -> IT:
        if self.env.unlabeled.empty:
            raise StopIteration
        self.update_sample()
        while self.current_sample:
            ins_key = self.current_sample.popleft()
            if ins_key not in self.env.labeled:
                return self.env.dataset[ins_key]
        if not self.env.unlabeled.empty:
            return self.__next__()
        raise StopIteration

    @classmethod
    def builder(
        cls,
        classifier_builder: Callable[
            [AbstractEnvironment[IT, KT, DT, VT, RT, str], il.LabelProvider[KT, str]],
            il.AbstractClassifier[
                IT, KT, DT, VT, RT, str, npt.NDArray[Any], npt.NDArray[Any]
            ],
        ],
        labeltransformer: AbstractLabelTransformer[KT, Any, LT],
        criteria_transformer: AbstractCriteriaSplitter[KT, str, str],
        batch_size: int = 1,
        chunk_size: int = 2000,
        identifier: Optional[str] = None,
    ) -> Callable[..., Self]:
        def builder_func(
            env: AbstractEnvironment[IT, KT, DT, VT, RT, str],
            pos_label: LT,
            neg_label: LT,
            *_,
            identifier: Optional[str] = identifier,
            **__,
        ):
            sublabels = criteria_transformer(env.labels)
            classifiers = value_map(lambda lbl: classifier_builder(env, lbl), sublabels)
            return cls(
                env,
                classifiers,
                labeltransformer,
                criteria_transformer,
                pos_label,
                neg_label,
                batch_size,
                chunk_size=chunk_size,
                identifier=identifier,
            )

        return builder_func


class MinSample(
    CriteriaTarLearner[IT, KT, DT, VT, RT, LT], Generic[IT, KT, DT, VT, RT, LT]
):
    def _rank(
        self, provider: il.InstanceProvider[IT, KT, DT, VT, RT]
    ) -> Sequence[Tuple[KT, float]]:
        predictions = self._predict(provider)
        labels = self._predict_stats(provider)

        def get_negs():
            for ins_key in provider:
                neg_preds = [
                    predictions[ck][ins_key]["False"]
                    for ck in self.criteria
                    if labels[ck].get_instances_by_label("True")
                    or labels[ck].get_instances_by_label("Unknown")
                ]
                yield ins_key, max(neg_preds)

        highest_negs = list(get_negs())
        ranking = sorted(highest_negs, key=lambda x: x[1])
        return ranking


class MinSampleExtended(
    CriteriaTarLearner[IT, KT, DT, VT, RT, LT], Generic[IT, KT, DT, VT, RT, LT]
):
    def _rank(
        self, provider: il.InstanceProvider[IT, KT, DT, VT, RT]
    ) -> Sequence[Tuple[KT, float]]:
        predictions = self._predict(provider)

        def get_negs():
            for ins_key in provider:
                neg_preds = [predictions[ck][ins_key]["False"] for ck in self.criteria]
                yield ins_key, np.mean(neg_preds)

        highest_negs = list(get_negs())
        ranking = sorted(highest_negs, key=lambda x: x[1])
        return ranking


class CriteriaWSA(
    CriteriaTarLearner[IT, KT, DT, VT, RT, LT],
    Generic[IT, KT, DT, VT, RT, LT],
):
    _name = "CriteriaWSA"
    wsa_classifier: CriteriaLLMClassifier[IT, KT, DT, VT, RT, LT, Any, Any, Any]
    agreement_classifiers: Mapping[
        str,
        il.AbstractClassifier[
            IT, KT, DT, VT, RT, Acceptance, npt.NDArray[Any], npt.NDArray[Any]
        ],
    ]
    answer_classifiers: Mapping[
        str,
        il.AbstractClassifier[
            IT, KT, DT, VT, RT, str, npt.NDArray[Any], npt.NDArray[Any]
        ],
    ]
    binary_classifier:  il.AbstractClassifier[
                IT, KT, DT, VT, RT, LT, npt.NDArray[Any], npt.NDArray[Any]
    ]
    subsetter: SubSetter
    proba_model: MockProbaClassifier
    combined_classifier: MockProbaClassifier
    wsa_predictions: il.LabelProvider[KT, str]
    acceptance_labels: Mapping[str, il.LabelProvider[KT, Acceptance]]
    pseudo_acceptance_labels: Mapping[str, il.LabelProvider[KT, Acceptance]]
    pseudo_labels: MutableMapping[str, il.LabelProvider[KT, str]]
    pseudo_labeled: MutableMapping[str, il.InstanceProvider[IT, KT, DT, VT, RT]]
    rejected: Mapping[str, il.InstanceProvider[IT, KT, DT, VT, RT]]
    acceptance_probas: MutableMapping[str, SimpleProbaProvider[KT, Acceptance]]

    def __init__(
        self,
        env: AbstractEnvironment[IT, KT, DT, VT, RT, str],
        classifiers: Mapping[
            str,
            il.AbstractClassifier[
                IT,
                KT,
                DT,
                VT,
                RT,
                str,
                npt.NDArray[Any],
                npt.NDArray[Any],
            ],
        ],
        wsa_classifier: CriteriaLLMClassifier[IT, KT, DT, VT, RT, LT, Any, Any, Any],
        agreement_classifiers: Mapping[
            str,
            il.AbstractClassifier[
                IT, KT, DT, VT, RT, Acceptance, npt.NDArray[Any], npt.NDArray[Any]
            ],
        ],
        answer_classifiers: Mapping[
            str,
            il.AbstractClassifier[
                IT, KT, DT, VT, RT, str, npt.NDArray[Any], npt.NDArray[Any]
            ],
        ],
        subsetter: SubSetter[KT],
        labeltransformer: AbstractLabelTransformer[KT, Any, LT],
        criteriatransformer: AbstractCriteriaSplitter[KT, Any, str],
        binary_classifier:  il.AbstractClassifier[
                IT, KT, DT, VT, RT, LT, npt.NDArray[Any], npt.NDArray[Any]
        ],
        pos_label: LT,
        neg_label: LT,
        batch_size: int,
        *_,
        seed: int = 0,
        chunk_size: int = 2000,
        identifier: str | None = None,
        **__,
    ) -> None:
        super().__init__(
            env,
            classifiers,
            labeltransformer,
            criteriatransformer,
            pos_label,
            neg_label,
            batch_size,
            *_,
            seed=seed,
            chunk_size=chunk_size,
            identifier=identifier,
            **__,
        )
        self.wsa_classifier = wsa_classifier
        self.wsa_predictions = il.MemoryLabelProvider.from_tuples(
            self.wsa_classifier.predict(self.env.dataset)
        )
        
        self.binary_classifier = binary_classifier
        self.acceptance_classifiers = agreement_classifiers
        self.answer_classifiers = answer_classifiers
        self.subsetter = subsetter
        self.pseudo_labels = {
            k: il.MemoryLabelProvider.from_provider(
                self.subsetter(self.env.labels, k), subset=[]
            )
            for k in self.criteria
        }
        self.pseudo_labeled = {
            k: self.env.create_empty_provider() for k in self.criteria
        }
        self._determine_acceptance()
        self.pseudo_acceptance_labels = {
            k: il.MemoryLabelProvider((Acceptance.ACCEPT, Acceptance.REJECT), dict())
            for k in self.criteria
        }
        self.acceptance_predictions = {
            k: il.MemoryLabelProvider((Acceptance.ACCEPT, Acceptance.REJECT), dict())
            for k in self.criteria
        }
        self.proba_model = MockProbaClassifier.from_provider(
            self.wsa_predictions, self.pos_label, self.neg_label
        )
        self.combined_classifier = MockProbaClassifier.from_provider(
            self.wsa_predictions, self.pos_label, self.neg_label
        )
        self.acceptance_probas = {k: SimpleProbaProvider([]) for k in self.criteria}
        self.last_ret_it = self.it
        self.reject_threshold = 0.4
        self.accept_threshold = 0.8
        self.k_sample = 100
        self.wsa_subsets = {ck: self.subsetter(self.wsa_predictions, ck) for ck in self.criteria}
        self.determine_excluded_keys(self.env.unlabeled)

    def _determine_acceptance(self):
        self.acceptance_labels = {
            ck: get_agreement_labels(
                self.env.labeled,
                self.subsetter(self.env.labels, ck),
                self.subsetter(self.wsa_predictions, ck),
            )
            for ck in self.criteria
        }
        self.rejected = {
            ck: self.env.create_bucket(lbl.get_instances_by_label(Acceptance.REJECT))
            for ck, lbl in self.acceptance_labels.items()
        }

    def determine_excluded_keys(self, provider: il.InstanceProvider[IT, KT, DT, VT, RT]):
        excluded_keys = {ck: clf.get_instances_by_label("False").intersection(provider) for ck,clf in self.wsa_subsets.items()}
        self.excluded_criteria_from_ranking = frozenset([ck for ck,exkeys in excluded_keys.items() if len(exkeys)/len(provider) >= 0.97])
        
    def set_as_labeled(self, instance: il.Instance[KT, DT, VT, RT]) -> None:
        super().set_as_labeled(instance)
        for k in self.criteria:
            self.pseudo_labeled[k].discard(instance)
        self.pseudo_labels = {
            k: il.MemoryLabelProvider.from_provider(v, self.pseudo_labeled[k].key_list)
            for k, v in self.pseudo_labels.items()
        }
        self._determine_acceptance()

    def _update_combined_classifier(self):
        if all([clf.fitted for clf in self.classifiers.values()]):
            classifier_probas = value_map(
                lambda clf: SimpleProbaProvider(clf.predict_proba(self.env.dataset)),
                self.classifiers,
            )
            classifier_preds = value_map(
                lambda clf: il.MemoryLabelProvider.from_tuples(
                    clf.predict(self.env.dataset)
                ),
                self.classifiers,
            )
            self.combined_classifier = MockProbaClassifier.from_classifiers(
                classifier_preds, classifier_probas, self.pos_label, self.neg_label
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
    
    def _update_proba_model(self):
        if all([clf.fitted for clf in self.answer_classifiers.values()]):
            classifier_probas = value_map(
                lambda clf: SimpleProbaProvider(clf.predict_proba(self.env.dataset)),
                self.classifiers,
            )
            classifier_preds = {
                ck: self.subsetter(self.wsa_predictions, ck) for ck in self.criteria
            }
            answer_preds = value_map(
                lambda clf: SimpleProbaProvider(clf.predict_proba(self.env.dataset)),
                self.answer_classifiers,
            )
            self.proba_model = MockProbaClassifier.from_corrections(
                self.wsa_predictions,
                classifier_preds,
                classifier_probas,
                self.acceptance_probas,
                answer_preds,
                self.pos_label,
                self.neg_label,
            )

    def _temp_augment_and_train(self):
        temp_labels = self.labeltransformer(self.env.labels)
        binary_llm = self.labeltransformer(self.wsa_predictions)
        llm_irrelevant = self.env.create_bucket(
            binary_llm.get_instances_by_label(self.neg_label)
        )
        sampled_non_relevant = self._provider_sample(llm_irrelevant)
        if PSEUDO_INS_PROVIDER in self.env:
            pseudo_docs = self.env[PSEUDO_INS_PROVIDER]
            for ins_key in pseudo_docs:
                temp_labels.set_labels(ins_key, self.pos_label)
        else:
            pseudo_docs = self.env.create_bucket([])
        for ins_key in sampled_non_relevant:
            if ins_key not in self.env.labeled:
                temp_labels.set_labels(ins_key, self.neg_label)
        train_set = self.env.combine(
            sampled_non_relevant, self.env.labeled, pseudo_docs
        )
        self.binary_classifier.fit_provider(train_set, temp_labels)


    def _fitandtrain(self):
        for ck in self.criteria:
            labels = self.subsetter(self.env.labels, ck)
            pseudo_labels = self.pseudo_labels[ck]
            temp_labels = combine_right_override(pseudo_labels, labels)
            train_set = self.env.create_bucket(
                frozenset(self.env.labeled)
                .union(self.pseudo_labeled[ck])
                .intersection(temp_labels.keys())
            )
            self.classifiers[ck].fit_provider(train_set, temp_labels)
            # Fit Answer Error Correction
            if check_fit_ability(self.rejected[ck], labels):
                self.answer_classifiers[ck].fit_provider(self.rejected[ck], labels)
        self._update_proba_model()

    def update_sample(self) -> Deque[KT]:
        if not self.current_sample:
            self.stats.update(self)  # type: ignore
            self._pseudo_label_provider(self.env.unlabeled)
            self._fitandtrain()
            self._update_combined_classifier()
            ranking = self._rank(self.env.unlabeled)
            sample = self._sample(ranking)

            # Store current sample
            self.current_sample = collections.deque(sample)

            # Record keeping
            self.rank_history[self.it] = self._to_history(ranking)
            self.sampled_sets[self.it] = tuple(sample)
            self.batch_sizes[self.it] = self.batch_size * 2

            self.it += 1
        return self.current_sample

    # def _rank(
    #     self, provider: il.InstanceProvider[IT, KT, DT, VT, RT]
    # ) -> Sequence[Tuple[KT, float]]:
    #     raw_probas = self.proba_model.predict_proba_provider_raw(provider)
    #     keys, matrix = raw_proba_chainer(raw_probas)
    #     pos_column = self.proba_model.get_label_column_index(self.pos_label)
    #     prob_vector = matrix[:, pos_column]
    #     floats: Sequence[float] = prob_vector.tolist()
    #     zipped = list(zip(keys, floats))
    #     ranking = sort_on(1, zipped)
    #     return ranking

    def _pseudo_label_provider(
        self, provider: il.InstanceProvider[IT, KT, DT, VT, RT]
    ) -> None:
        for k in self.criteria:
            self.acceptance_classifiers[k].fit_provider(
                self.env.labeled, self.acceptance_labels[k]
            )
            self.acceptance_predictions = value_map(
                lambda clf: il.MemoryLabelProvider.from_tuples(
                    clf.predict(self.env.dataset)
                ),
                self.acceptance_classifiers,
            )
            probas = self.acceptance_classifiers[k].predict_proba(provider)
            self.acceptance_probas[k] = SimpleProbaProvider(probas)
            # Add the top 5% percent of ACCEPTANCE, or above 75 %
            top_5_proba = self.acceptance_probas[k].quantile(Acceptance.ACCEPT, 0.95)
            strong_accepts = self.env.create_bucket(
                self.acceptance_probas[k].above(
                    Acceptance.ACCEPT, max(0.75, top_5_proba)
                )
            )
            # strong_rejects = self.env.create_bucket(proba_prov.above(Acceptance.REJECT, 0.70))
            # combined = self.env.combine(strong_accepts, strong_rejects)
            self.pseudo_labels[k] = self.subsetter(
                il.MemoryLabelProvider.from_tuples(
                    self.wsa_classifier.predict(strong_accepts)
                ),
                k,
            )
            self.pseudo_labeled[k] = strong_accepts

    @classmethod
    def builder(
        cls,
        classifier_builder: Callable[
            [AbstractEnvironment[IT, KT, DT, VT, RT, str], il.LabelProvider[KT, str]],
            il.AbstractClassifier[
                IT, KT, DT, VT, RT, str, npt.NDArray[Any], npt.NDArray[Any]
            ],
        ],
        wsa_classifier: CriteriaLLMClassifier[IT, KT, DT, VT, RT, LT, Any, Any, Any],
        acceptance_builder: Callable[
            [AbstractEnvironment[IT, KT, DT, VT, RT, Any]],
            il.AbstractClassifier[
                IT, KT, DT, VT, RT, Acceptance, npt.NDArray[Any], npt.NDArray[Any]
            ],
        ],
        corrector_builder: Callable[
            [AbstractEnvironment[IT, KT, DT, VT, RT, str], il.LabelProvider[KT, str]],
            il.AbstractClassifier[
                IT, KT, DT, VT, RT, str, npt.NDArray[Any], npt.NDArray[Any]
            ],
        ],
        binary_builder:  Callable[
            [AbstractEnvironment[IT, KT, DT, VT, RT, str], il.LabelProvider[KT, str]],
            il.AbstractClassifier[
                IT, KT, DT, VT, RT, str, npt.NDArray[Any], npt.NDArray[Any]
            ],
        ],
        subsetter: SubSetter[KT],
        labeltransformer: AbstractLabelTransformer[KT, Any, LT],
        criteria_transformer: AbstractCriteriaSplitter[KT, str, str],
        batch_size: int = 1,
        chunk_size: int = 2000,
        identifier: Optional[str] = None,
    ) -> Callable[..., Self]:
        def builder_func(
            env: AbstractEnvironment[IT, KT, DT, VT, RT, str],
            pos_label: LT,
            neg_label: LT,
            *_,
            identifier: Optional[str] = identifier,
            **__,
        ):
            sublabels = criteria_transformer(env.labels)
            criteria = list(sublabels.keys())
            classifiers = value_map(lambda lbl: classifier_builder(env, lbl), sublabels)
            binary_classifier = binary_builder(env, labeltransformer(env.labels)) # type: ignore
            answer_envs = {
                ck: wsa_classifier.get_answer_env(
                    ck, env, [Acceptance.ACCEPT, Acceptance.REJECT]
                )
                for ck in criteria
            }
            target_set = list(subsetter(env.labels, criteria[0]).labelset)
            answer_envs_error = {
                ck: wsa_classifier.get_answer_env(ck, env, target_set)
                for ck in criteria
            }
            acceptance_classifiers = value_map(acceptance_builder, answer_envs)
            answer_correctors = {
                ck: corrector_builder(a_env, sublabels[ck])
                for ck, a_env in answer_envs_error.items()
            }
            return cls(
                env,
                classifiers,
                wsa_classifier,
                acceptance_classifiers,
                answer_correctors,
                subsetter,
                labeltransformer,
                criteria_transformer,
                binary_classifier,
                pos_label,
                neg_label,
                batch_size,
                chunk_size=chunk_size,
                identifier=identifier,
            )

        return builder_func


class SkipNegWSA(CriteriaWSA[IT, KT, DT, VT, RT, LT], Generic[IT, KT, DT, VT, RT, LT]):
    def _fitpredict_acceptance(self) -> None:
        for k in self.criteria:
            self.acceptance_classifiers[k].fit_provider(
                self.env.labeled, self.acceptance_labels[k]
            )
        self.acceptance_predictions = value_map(
            lambda clf: il.MemoryLabelProvider.from_tuples(
                clf.predict(self.env.dataset)
            ),
            self.acceptance_classifiers,
        )
        self.acceptance_probas = value_map(
            lambda clf: SimpleProbaProvider(clf.predict_proba(self.env.dataset)),
            self.acceptance_classifiers,
        )  # type: ignore

    def skippable(self, ins: il.Instance[KT, Any, Any, Any]) -> bool:
        key = to_key(ins)
        wsa_preds = labels_to_symbols(self.wsa_predictions[key])
        
        neg_accept_preds = [
            self.acceptance_probas[ck].proba(key, Acceptance.ACCEPT)
            for ck, status in wsa_preds.items()
            if not weakly(status) and ck not in self.excluded_criteria_from_ranking
        ]
        if neg_accept_preds:
            highest_accept = max(neg_accept_preds)
            if highest_accept >= self.accept_threshold:
                return True
        # min_reject = min([self.acceptance_probas[k].proba(key, Acceptance.REJECT) for k in self.criteria])
        # if min_reject >= self.reject_threshold:
        #     return False
        return False

    def _pseudo_label_instance(self, ins: il.Instance[KT, DT, VT, RT]) -> None:
        key = to_key(ins)
        wsa_labels = labels_to_symbols(self.wsa_predictions[ins])
        for ck, status in wsa_labels.items():
            if (
                self.acceptance_probas[ck].proba(key, Acceptance.ACCEPT)
                >= self.accept_threshold
            ):
                self.pseudo_labels[ck].set_labels(key, str(status))
                self.pseudo_acceptance_labels[ck].set_labels(key, Acceptance.ACCEPT)
                self.pseudo_labeled[ck].add(self.env.dataset[key])

    def remove_previous_round(
        self, provider: il.InstanceProvider[IT, KT, DT, VT, RT]
    ) -> il.InstanceProvider[IT, KT, DT, VT, RT]:
        if self.it >= 1:
            previous_round: Sequence[KT] = [
                k
                for it in range(max(0, self.last_ret_it), max(0, self.it - 1))
                for k in self.sampled_sets[it]
            ]
        else:
            previous_round: Sequence[KT] = tuple()
        prvkeys = frozenset(provider)
        exclude_previous_round = prvkeys.difference(previous_round)
        new_provider = self.env.create_bucket(exclude_previous_round)
        return new_provider

    def update_sample(self) -> Deque[KT]:
        if not self.current_sample:
            self.stats.update(self)  # type: ignore
            self._fitpredict_acceptance()
            self._fitandtrain()
            self._temp_augment_and_train()
            self._update_combined_classifier()
            self._update_proba_model()
            self.determine_excluded_keys(self.env.unlabeled)
            ranking = self._rank(self.env.unlabeled)
            wsa_ranking = self._rank_wsa(self.env.unlabeled)
            binary_ranking = self._rank_binary(self.env.unlabeled)
            sample = self._sample_multiple(binary_ranking, wsa_ranking)

            # Store current sample
            self.current_sample = collections.deque(sample)

            # Record keeping
            self.rank_history[self.it] = self._to_history(binary_ranking)
            self.sampled_sets[self.it] = tuple(sample)
            self.batch_sizes[self.it] = self.batch_size * 3

            self.it += 1
        return self.current_sample

    def __next__(self) -> IT:
        if self.env.unlabeled.empty:
            raise StopIteration
        self.update_sample()
        while self.current_sample:
            ins_key = self.current_sample.popleft()
            if ins_key not in self.env.labeled:
                ins = self.env.dataset[ins_key]
                # An instance should have a 10% chance to be selected
                # regardless of its status
                chance = random.uniform(0, 1)
                if chance > 0.90 or not self.skippable(ins):
                    self.last_ret_it = self.it
                    return ins
                self._pseudo_label_instance(ins)
        if not self.env.unlabeled.empty:
            return next(self)
        raise StopIteration

    def _sample_multiple(
        self, *distributions: Sequence[Tuple[KT, float]]
    ) -> Sequence[KT]:
        samples = list()
        for distribution in distributions:
            keys, _ = list_unzip(distribution)
            sample = keys[0 : self.batch_size]
            samples.extend(sample)
        return samples

    def _rank(
        self, provider: il.InstanceProvider[IT, KT, DT, VT, RT]
    ) -> Sequence[Tuple[KT, float]]:
        predictions = self._predict(provider)
        labels = self._predict_stats(provider)

        def get_negs():
            for ins_key in provider:
                # wsa_preds = labels_to_symbols(self.wsa_predictions[ins_key])
                neg_preds = [
                    predictions[ck][ins_key]["False"]
                    for ck in self.criteria
                    if labels[ck].get_instances_by_label("True")
                    or labels[ck].get_instances_by_label("Unknown")
                ]
                yield ins_key, max(neg_preds)

        highest_negs = list(get_negs())
        ranking = sorted(highest_negs, key=lambda x: x[1])
        return ranking

    def _rank_combined(
        self, provider: il.InstanceProvider[IT, KT, DT, VT, RT]
    ) -> Sequence[Tuple[KT, float]]:
        raw_probas = self.combined_classifier.predict_proba_provider_raw(provider)
        keys, matrix = raw_proba_chainer(raw_probas)
        pos_column = self.combined_classifier.get_label_column_index(self.pos_label)
        prob_vector = matrix[:, pos_column]
        floats: Sequence[float] = prob_vector.tolist()
        zipped = list(zip(keys, floats))
        ranking = sort_on(1, zipped)
        return ranking

    def _rank_wsa(
        self, provider: il.InstanceProvider[IT, KT, DT, VT, RT]
    ) -> Sequence[Tuple[KT, float]]:
        raw_probas = self.proba_model.predict_proba_provider_raw(provider)
        keys, matrix = raw_proba_chainer(raw_probas)
        pos_column = self.proba_model.get_label_column_index(self.pos_label)
        prob_vector = matrix[:, pos_column]
        floats: Sequence[float] = prob_vector.tolist()
        zipped = list(zip(keys, floats))
        ranking = sort_on(1, zipped)
        return ranking
    
    def _rank_binary(
        self, provider: il.InstanceProvider[IT, KT, DT, VT, RT]
    ) -> Sequence[Tuple[KT, float]]:
        raw_probas = self.binary_classifier.predict_proba_provider_raw(provider)
        keys, matrix = raw_proba_chainer(raw_probas)
        pos_column = self.binary_classifier.get_label_column_index(self.pos_label)
        prob_vector = matrix[:, pos_column]
        floats: Sequence[float] = prob_vector.tolist()
        zipped = list(zip(keys, floats))
        ranking = sort_on(1, zipped)
        return ranking
