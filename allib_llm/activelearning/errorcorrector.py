import collections
from math import ceil
from typing import Any, Callable, Deque, Generic, Optional, Sequence, Tuple

import instancelib as il
import numpy.typing as npt
from allib.activelearning.autotar import BinaryTarLearner
from allib.activelearning.autotarensemble import add_doc
from allib.activelearning.base import ActiveLearner
from allib.activelearning.learnersequence import LearnerSequence
from allib.analysis.base import StatsMixin
from allib.environment.base import AbstractEnvironment
from allib.stopcriterion.base import AbstractStopCriterion
from allib.typehints.typevars import DT, IT, KT, LT, RT, VT
from cleanlab.classification import CleanLearning
from instancelib.utils.func import sort_on
from sklearn.linear_model import LogisticRegression
from typing_extensions import Self
from allib.machinelearning.taroptimized import ALSklearn

from ..machinelearning.labelcorrection import (
    noisy_label_correction,
    find_likely_relevant,
)
from allib.machinelearning.sparse import SparseVectorStorage


def transfer(
    a: ActiveLearner[IT, KT, DT, VT, RT, LT], b: ActiveLearner[IT, KT, DT, VT, RT, LT]
) -> None:
    for ins in a.env.labeled:
        add_doc(b, ins, *a.env.labels[ins])


class TwoPhase(LearnerSequence[IT, KT, DT, VT, RT, LT]):
    _name = "LLM+CAL"

    def __init__(
        self,
        env: AbstractEnvironment[IT, KT, DT, VT, RT, LT],
        learners: Sequence[ActiveLearner[IT, KT, DT, VT, RT, LT],],
        stopcriteria: Sequence[AbstractStopCriterion[LT]],
        pos_label: LT,
        neg_label: LT,
        *_,
        identifier: Optional[str] = None,
        **__,
    ) -> None:
        super().__init__(env, learners, stopcriteria, *_, identifier=identifier, **__)
        self.pos_label = pos_label
        self.neg_label = neg_label

    def _choose_learner(self) -> ActiveLearner[IT, KT, DT, VT, RT, LT]:
        learner = super()._choose_learner()
        if self.current_learner == 1 and not learner.env.labeled:
            transfer(self.learners[0], self.learners[1])
        return learner

    @classmethod
    def builder(
        cls,
        llm_builder: Callable[..., ActiveLearner[IT, KT, DT, VT, RT, LT]],
        tar_builder: Callable[..., ActiveLearner[IT, KT, DT, VT, RT, LT]],
        llm_criterion: Callable[[LT, LT], AbstractStopCriterion[LT]],
        *_: Any,
        **__: Any,
    ) -> Callable[..., Self]:
        learner_builders = [
            llm_builder,
            tar_builder,
        ]

        def wrap_func(
            env: AbstractEnvironment[IT, KT, DT, VT, RT, LT],
            pos_label: LT,
            neg_label: LT,
            *args,
            **kwargs,
        ):
            stop_criteria = [
                llm_criterion(pos_label, neg_label),
            ]
            learners = [
                builder(
                    env.from_environment(env),
                    pos_label=pos_label,
                    neg_label=neg_label,
                    *args,
                    **kwargs,
                )
                for builder in learner_builders
            ]
            return cls(env, learners, stop_criteria, pos_label, neg_label)

        return wrap_func


class ErrorCorrector(
    BinaryTarLearner[IT, KT, DT, VT, RT, LT],
    StatsMixin[KT, LT],
    Generic[IT, KT, DT, VT, RT, LT],
):
    _name = "ErrorCorrector"
    trained_classifier: il.AbstractClassifier[
        IT, KT, DT, VT, RT, LT, npt.NDArray[Any], npt.NDArray[Any]
    ]
    predictions: il.LabelProvider[KT, LT]
    normal: bool

    def __init__(
        self,
        env: AbstractEnvironment[IT, KT, DT, VT, RT, LT],
        classifier: il.AbstractClassifier[
            IT, KT, DT, VT, RT, LT, npt.NDArray[Any], npt.NDArray[Any]
        ],
        trained_classifier: il.AbstractClassifier[
            IT, KT, DT, VT, RT, LT, npt.NDArray[Any], npt.NDArray[Any]
        ],
        pos_label: LT,
        neg_label: LT,
        batch_size: int,
        *_,
        seed: int = 0,
        chunk_size: int = 2000,
        identifier: Optional[str] = None,
        **__,
    ) -> None:
        super().__init__(
            env,
            classifier,
            pos_label,
            neg_label,
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
        return self.env.create_bucket(
            self.predictions.get_instances_by_label(self.pos_label).intersection(
                self.env.unlabeled
            )
        )

    def _rank_by_trained(
        self, provider: il.InstanceProvider[IT, KT, DT, VT, RT]
    ) -> Sequence[Tuple[KT, float]]:
        keys, matrix = self._predict(provider)
        pos_column = self.classifier.get_label_column_index(self.pos_label)
        prob_vector = matrix[:, pos_column]
        floats: Sequence[float] = prob_vector.tolist()
        zipped = list(zip(keys, floats))
        ranking = sort_on(1, zipped)
        return ranking

    def update_sample(self) -> Deque[KT]:
        if not self.current_sample:
            self.stats.update(self)

            candidates = self.get_candidate_set()
            self.classifier.fit_provider(self.env.dataset, self.predictions)
            if candidates:
                ranking = self._rank_by_trained(candidates)
            else:
                ranking = self._rank_by_trained(self.env.unlabeled)
            sample = self._sample(ranking)

            # Store current sample
            self.current_sample = collections.deque(sample)

            # Record keeping
            self.rank_history[self.it] = self._to_history(ranking)
            self.sampled_sets[self.it] = tuple(sample)
            self.batch_sizes[self.it] = self.batch_size

            # Increment batch_size for next for train iteration
            if self.normal:
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
        k_sample: int,
        batch_size: int,
        chunk_size: int = 2000,
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
            classifier = classifier_builder(env)
            trained_classifier = trained_classifier_builder(env)
            return cls(
                env,
                classifier,
                trained_classifier,
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
        k_sample: int,
        batch_size: int,
        chunk_size: int = 2000,
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
            classifier = classifier_builder(env)
            return cls(
                env,
                classifier,
                trained_classifier,
                pos_label,
                neg_label,
                k_sample,
                batch_size,
                chunk_size=chunk_size,
                identifier=identifier,
            )

        return builder_func


class ErrorCorrector2(
    ErrorCorrector[IT, KT, DT, VT, RT, LT], Generic[IT, KT, DT, VT, RT, LT]
):
    def update_sample(self) -> Deque[KT]:
        if not self.current_sample:
            self.stats.update(self)

            candidates = self.get_candidate_set()
            try:
                self.classifier.fit_provider(self.env.labeled, self.env.labels)
            except:
                self.classifier.fit_provider(self.env.dataset, self.predictions)
            if candidates:
                ranking = self._rank_by_trained(candidates)
            else:
                ranking = self._rank_by_trained(self.env.unlabeled)
            sample = self._sample(ranking)

            # Store current sample
            self.current_sample = collections.deque(sample)

            # Record keeping
            self.rank_history[self.it] = self._to_history(ranking)
            self.sampled_sets[self.it] = tuple(sample)
            self.batch_sizes[self.it] = self.batch_size

            # Increment batch_size for next for train iteration
            if self.normal:
                self.batch_size += ceil(self.batch_size / 10)

            self.it += 1
        return self.current_sample


class ErrorCorrector3(
    ErrorCorrector[IT, KT, DT, VT, RT, LT], Generic[IT, KT, DT, VT, RT, LT]
):
    def __init__(
        self,
        env: AbstractEnvironment[IT, KT, DT, VT, RT, LT],
        classifier: il.AbstractClassifier[
            IT, KT, DT, VT, RT, LT, npt.NDArray[Any], npt.NDArray[Any]
        ],
        trained_classifier: il.AbstractClassifier[
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
            classifier,
            trained_classifier,
            pos_label,
            neg_label,
            batch_size,
            *_,
            seed=seed,
            chunk_size=chunk_size,
            identifier=identifier,
            **__,
        )
        self.lr = LogisticRegression()
        self.cl = CleanLearning(self.lr, cv_n_folds=5)
        self.insclf = ALSklearn.build(
            self.cl, self.env, self.classifier.vectorizer, SparseVectorStorage  # type: ignore
        )  # type: ignore

    def get_candidate_set(self) -> il.InstanceProvider[IT, KT, DT, VT, RT]:
        corrected_candidates = find_likely_relevant(
            self.env,
            self.predictions,
            self.pos_label,
            self.neg_label,
            self.cl,
            self.insclf,  # type: ignore
        )
        unlabeled_candidates = self.env.create_bucket(
            frozenset(corrected_candidates).intersection(self.env.unlabeled)
        )
        return unlabeled_candidates


class ErrorCorrector4(
    ErrorCorrector3[IT, KT, DT, VT, RT, LT], Generic[IT, KT, DT, VT, RT, LT]
):
    def update_sample(self) -> Deque[KT]:
        if not self.current_sample:
            self.stats.update(self)

            candidates = self.get_candidate_set()
            try:
                self.classifier.fit_provider(self.env.labeled, self.env.labels)
            except:
                self.classifier.fit_provider(self.env.dataset, self.predictions)
            if candidates:
                ranking = self._rank_by_trained(candidates)
            else:
                ranking = self._rank_by_trained(self.env.unlabeled)
            sample = self._sample(ranking)

            # Store current sample
            self.current_sample = collections.deque(sample)

            # Record keeping
            self.rank_history[self.it] = self._to_history(ranking)
            self.sampled_sets[self.it] = tuple(sample)
            self.batch_sizes[self.it] = self.batch_size

            # Increment batch_size for next for train iteration
            if self.normal:
                self.batch_size += ceil(self.batch_size / 10)

            self.it += 1
        return self.current_sample
