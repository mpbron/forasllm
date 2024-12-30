from typing import FrozenSet, Iterator, Mapping, Sequence, Tuple, TypeVar
import instancelib as il
from instancelib.typehints.typevars import LT, KT
import numpy as np
from instancelib.utils.func import list_unzip

_T = TypeVar("_T")


def binary_error_rate(
    pos_label: LT, pos_precision: float, neg_label: LT, neg_precision: float
) -> Mapping[LT, Mapping[LT, float]]:
    return {
        pos_label: {pos_label: pos_precision, neg_label: 1.0 - pos_precision},
        neg_label: {pos_label: 1.0 - neg_precision, neg_label: neg_precision},
    }


def mock_errors(
    source: il.LabelProvider[KT, LT],
    target_distribution: Mapping[LT, Mapping[_T, float]],
    rng: np.random.Generator = np.random.default_rng(),
) -> il.MemoryLabelProvider[KT, _T]:
    def yield_results() -> Iterator[Tuple[KT, FrozenSet[_T]]]:
        for lbl_name in source.labelset:
            inss = list(source.get_instances_by_label(lbl_name))
            target_labels_ps = target_distribution[lbl_name]

            target_labels, target_ps = list_unzip(target_labels_ps.items())
            n = len(target_ps)
            indices = np.arange(n)
            selected_indices = rng.choice(indices, size=len(inss), p=target_ps)
            for x, index in zip(inss, selected_indices):
                yield (x, frozenset([target_labels[index]]))

    results = list(yield_results())
    provider = il.MemoryLabelProvider.from_tuples(results)
    return provider
