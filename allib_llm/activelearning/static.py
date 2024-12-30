import collections
import logging
import random
from typing import Any, Callable, Dict, Generic, Optional, Sequence, Tuple

import instancelib as il
from allib.activelearning.poolbased import PoolBasedAL
from allib.environment.base import AbstractEnvironment
from allib.typehints.typevars import DT, IT, KT, LT, RT, VT
from typing_extensions import Self

from ..utils.instances import (
    get_instances_with_labels,
    get_subset_by_labels_intersection,
)

LOGGER = logging.getLogger(__name__)

# TODO make this generic without constants
A_LABELS = frozenset({"a_Y", "b_Y", "c_Y", "d_Y"})
Q_LABELS = frozenset({"a_Q", "b_Q", "c_Q", "d_Q"})
R_LABELS = frozenset({"a_N", "b_N", "c_N", "d_N"})


class FixedOrdering(
    PoolBasedAL[IT, KT, DT, VT, RT, LT], Generic[IT, KT, DT, VT, RT, LT]
):
    _name = "FixedOrdering"

    def __init__(
        self,
        env: AbstractEnvironment[IT, KT, DT, VT, RT, LT],
        ordering: Sequence[KT],
        label: LT,
        *_,
        identifier: Optional[str] = None,
        **__,
    ) -> None:
        super().__init__(env, identifier=identifier)
        self.ordering = collections.deque(ordering)
        self.label = label

    @property
    def name(self) -> Tuple[str, Optional[LT]]:
        if self.identifier is not None:
            return f"{self.identifier}", self.label
        return f"{self._name}", self.label

    def update_ordering(self) -> bool:
        return True

    @classmethod
    def builder(
        cls,
        model: il.AbstractClassifier[IT, KT, DT, VT, RT, LT, Any, Any],
        identifier: Optional[str] = None,
        probable_sample_size=50,
    ) -> Callable[[AbstractEnvironment[IT, KT, DT, VT, RT, LT], LT, LT], Self]:
        def build(
            env: AbstractEnvironment[IT, KT, DT, VT, RT, LT],
            pos_label: LT,
            neg_label: LT,
        ) -> Self:
            preds = model.predict(env.unlabeled)
            pred_provider = il.MemoryLabelProvider.from_tuples(preds)
            llm_labeled = get_instances_with_labels(env, env.unlabeled, labelprovider=pred_provider)
            direct_accept = get_subset_by_labels_intersection(
                env, env.unlabeled, *A_LABELS, labelprovider=pred_provider  # type: ignore
            )
            probable = env.create_bucket(
                frozenset(llm_labeled).difference(
                    env.get_subset_by_labels(
                        llm_labeled, *R_LABELS, labelprovider=pred_provider  # type: ignore
                    )
                ).intersection(env.unlabeled)
            )
            probable_sample = random.sample(probable.key_list, probable_sample_size)
            selection = (*direct_accept.key_list, *(s for s in probable_sample if s not in direct_accept))
            return cls(env, selection, pos_label, identifier=identifier)

        return build
