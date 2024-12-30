import collections
from typing import Any, Callable, Deque, FrozenSet, Generic, List, Mapping, Sequence

from allib.activelearning.base import ActiveLearner
from allib.stopcriterion.base import AbstractStopCriterion
from allib.stopcriterion.heuristic import AprioriRecallTarget
from allib.stopcriterion.others import AnnotationStatisticsSlim, BudgetStoppingRule
from instancelib.typehints.typevars import KT, LT
from instancelib.utils.func import all_equal, value_map
from typing_extensions import Self

from ..analysis.stats import CriteriaStatisticsSlim
from ..utils.instances import AbstractLabelTransformer, LabelTransformer


class AprioriCriteriaRecallTarget(AprioriRecallTarget[LT]):
    def __init__(
        self,
        pos_label: LT,
        neg_label: LT,
        label_transformer: LabelTransformer[KT, LT],
        target: float = 0.95,
    ):
        super().__init__(pos_label, target)
        self.label_transformer = label_transformer
        self.neg_label = neg_label

    def update(self, learner: ActiveLearner[Any, Any, Any, Any, Any, LT]) -> None:
        binary_labels = self.label_transformer(
            learner.env.labels, self.pos_label, self.neg_label
        )
        binary_truth = self.label_transformer(
            learner.env.truth, self.pos_label, self.neg_label
        )
        if not self.stopped:
            n_pos_truth = binary_truth.document_count(self.pos_label)
            n_pos_now = binary_labels.document_count(self.pos_label)
            self.stopped = n_pos_now / n_pos_truth >= self.target


class BudgetCriteriaStoppingRule(BudgetStoppingRule[KT, LT]):
    labeltransformer: LabelTransformer[KT, LT]

    def __init__(self, pos_label: LT, neg_label: LT, labeltransformer: LabelTransformer[KT, LT], target_size: int = 10) -> None:
        super().__init__(pos_label, target_size)
        self.neg_label = neg_label
        self.stats = CriteriaStatisticsSlim(labeltransformer, pos_label, neg_label)

class SameStateCount(AbstractStopCriterion[LT], Generic[KT, LT]):
    labeltransformer: AbstractLabelTransformer[KT, Any, LT]

    def __init__(self, labeltransformer: AbstractLabelTransformer[KT, Any, LT], label: LT, same_state_count: int):
        self.pos_label = label
        self.same_state_count = same_state_count
        self.pos_history: Deque[int] = collections.deque()
        self.has_been_different = False
        self.labeltransformer = labeltransformer

    def update(self, learner: ActiveLearner[Any, Any, Any, Any, Any, LT]):
        labels = self.labeltransformer(learner.env.labels)
        pos_instances = learner.env.get_subset_by_labels(learner.env.labeled, self.pos_label, labelprovider=labels)
        self.add_count(len(pos_instances))

    def add_count(self, value: int) -> None:
        if len(self.pos_history) > self.same_state_count:
            self.pos_history.pop()
        if self.pos_history and not self.has_been_different:
            previous_value = self.pos_history[0]
            if previous_value != value:
                self.has_been_different = True
        self.pos_history.appendleft(value)

    @property
    def count(self) -> int:
        return self.pos_history[0]

    @property
    def same_count(self) -> bool:
        return all_equal(self.pos_history)

    @property
    def stop_criterion(self) -> bool:
        if len(self.pos_history) < self.same_state_count:
            return False
        return self.has_been_different and self.same_count
    
    @classmethod
    def builder(cls, labeltransformer: AbstractLabelTransformer[KT, Any, LT], k: int) -> Callable[[LT, LT], Self]:
        def builder_func(pos_label: LT, neg_label: LT) -> Self:
            return cls(labeltransformer, pos_label, k)
        return builder_func
