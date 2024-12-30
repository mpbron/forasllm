from typing import Any, Callable, TypeVar

import instancelib as il
from allib.activelearning.base import ActiveLearner
from allib.analysis.analysis import BinaryPerformance
from allib.typehints import KT, LT


def process_performance(
    learner: ActiveLearner[Any, KT, Any, Any, Any, LT],
    pos_label: LT,
    neg_label: LT,
    label_transformer: Callable[
        [il.LabelProvider[KT, Any], LT, LT], il.LabelProvider[KT, LT]
    ],
) -> BinaryPerformance[KT, LT]:
    binary_labels = label_transformer(learner.env.labels, pos_label, neg_label)
    binary_truth = label_transformer(learner.env.truth, pos_label, neg_label)
    labeled = frozenset(learner.env.labeled)
    labeled_positives = frozenset(
        learner.env.get_subset_by_labels(
            learner.env.labeled, pos_label, labelprovider=binary_labels
        )
    )
    labeled_negatives = labeled.difference(labeled_positives)
    truth_positives = binary_truth.get_instances_by_label(pos_label)

    unlabeled = frozenset(learner.env.unlabeled)
    unlabeled_positives = unlabeled.intersection(
        binary_truth.get_instances_by_label(pos_label)
    )
    unlabeled_negatives = unlabeled.difference(unlabeled_positives)

    true_positives = labeled_positives.intersection(truth_positives)
    false_positives = labeled_positives.difference(truth_positives).union(
        labeled_negatives
    )
    false_negatives = truth_positives.difference(labeled_positives)
    true_negatives = unlabeled_negatives

    return BinaryPerformance(
        pos_label,
        None,
        true_positives,
        true_negatives,
        false_positives,
        false_negatives,
    )