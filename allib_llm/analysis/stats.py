from typing import Any, FrozenSet, List, Mapping, Sequence

from allib.activelearning.base import ActiveLearner
from allib.analysis.base import AnnotationStatisticsSlim
from instancelib.typehints.typevars import KT, LT
from instancelib.utils.func import value_map

from ..utils.instances import AbstractLabelTransformer, LabelTransformer


class CriteriaStatisticsSlim(AnnotationStatisticsSlim[KT, LT]):
    label_transformer: LabelTransformer[KT, LT]
    pos_label: LT
    neg_label: LT

    def __init__(self, labeltransformer: LabelTransformer[KT, LT], pos_label: LT, neg_label: LT) -> None:
        super().__init__()
        self.label_transformer = labeltransformer
        self.pos_label = pos_label
        self.neg_label = neg_label

    def update(self, learner: ActiveLearner[Any, KT, Any, Any, Any, LT]):
        binary_labels = self.label_transformer(learner.env.labels, self.pos_label, self.neg_label)
        annotated = frozenset(learner.env.labeled)
        unlabeled = frozenset(learner.env.unlabeled)
        if learner.env.labeled:
            current_round = {
                label: (
                    frozenset(
                        learner.env.get_subset_by_labels(
                            learner.env.labeled, label, labelprovider=binary_labels
                        )
                    )
                )
                for label in binary_labels.labelset
            }
            current_round_new = {
                label: keys.difference(self.previous_round)
                for label, keys in current_round.items()
            }
            current_round_counts = value_map(len, current_round)
            current_round_new_counts = value_map(len, current_round_new)
            annotated_new = annotated.difference(self.previous_round)
            self.annotated_per_round.append(len(annotated_new))
            self.labelwise.append(current_round_counts)
            self.annotated.append(len(annotated))
            self.unlabeled.append(len(unlabeled))
            self.per_round.append(current_round_new_counts)
            self.dataset.append(len(learner.env.dataset))
            self.previous_round = annotated


class CriteriaStatisticsSlimV2(AnnotationStatisticsSlim[KT, LT]):
    label_transformer: AbstractLabelTransformer[KT, Any, LT]
    
    def __init__(self, labeltransformer: AbstractLabelTransformer[KT, Any, LT]) -> None:
        super().__init__()
        self.label_transformer = labeltransformer

    def update(self, learner: ActiveLearner[Any, KT, Any, Any, Any, LT]):
        binary_labels = self.label_transformer(learner.env.labels)
        annotated = frozenset(learner.env.labeled)
        unlabeled = frozenset(learner.env.unlabeled)
        if learner.env.labeled:
            current_round = {
                label: (
                    frozenset(
                        learner.env.get_subset_by_labels(
                            learner.env.labeled, label, labelprovider=binary_labels
                        )
                    )
                )
                for label in binary_labels.labelset
            }
            current_round_new = {
                label: keys.difference(self.previous_round)
                for label, keys in current_round.items()
            }
            current_round_counts = value_map(len, current_round)
            current_round_new_counts = value_map(len, current_round_new)
            annotated_new = annotated.difference(self.previous_round)
            self.annotated_per_round.append(len(annotated_new))
            self.labelwise.append(current_round_counts)
            self.annotated.append(len(annotated))
            self.unlabeled.append(len(unlabeled))
            self.per_round.append(current_round_new_counts)
            self.dataset.append(len(learner.env.dataset))
            self.previous_round = annotated