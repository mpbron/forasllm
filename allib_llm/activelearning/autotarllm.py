import collections
from math import ceil
from typing import Any, Callable, Deque, Generic, Optional, Sequence, Tuple

import instancelib as il
import numpy.typing as npt
from allib.activelearning.autotar import AutoTarLearner, PSEUDO_INS_PROVIDER, pseudo_from_metadata
from allib.environment.base import AbstractEnvironment
from allib.typehints.typevars import DT, IT, KT, LT, RT, VT
from instancelib.utils.func import sort_on, list_unzip
from typing_extensions import Self
from typing import Mapping


class AutoTarOnLLM(
    AutoTarLearner[IT, KT, DT, VT, RT, LT],
    Generic[IT, KT, DT, VT, RT, LT],
):
    _name = "AutoTarOnLLM"
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
        temp_labels = il.MemoryLabelProvider[KT, LT].from_provider(self.env.labels)
        llm_irrelevant = self.env.create_bucket(
            self.predictions.get_instances_by_label(self.neg_label)
        )
        sampled_non_relevant = self._provider_sample(llm_irrelevant)
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
                pos_label,
                neg_label,
                k_sample,
                batch_size,
                chunk_size=chunk_size,
                identifier=identifier,
            )

        return builder_func
