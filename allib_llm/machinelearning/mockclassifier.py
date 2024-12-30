from __future__ import annotations

import itertools
import math
from typing import (
    Any,
    Callable,
    FrozenSet,
    Generic,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    Tuple,
    TypeVar,
)

import numpy as np
import numpy.typing as npt
from allib.environment.base import AbstractEnvironment
from instancelib.instances.base import Instance, InstanceProvider
from instancelib.labels.base import LabelProvider
from instancelib.labels.encoder import LabelEncoder, MultilabelDictionaryEncoder
from instancelib.machinelearning.base import AbstractClassifier
from instancelib.typehints import DT, KT, LMT, LT, LVT, PMT, RT, VT
from instancelib.utils.chunks import divide_iterable_in_lists
from instancelib.utils.func import invert_mapping, seq_or_map_to_map
from tqdm.auto import tqdm
from typing_extensions import Self

IT = TypeVar("IT", bound="Instance[Any, Any, Any, Any]")


from instancelib.machinelearning.wrapper import numpy_ova_threshold


def key_chunker(
    instances: Iterable[Instance[KT, Any, Any, Any]], batch_size: int = 200
) -> Iterator[Sequence[KT]]:
    chunks = divide_iterable_in_lists(instances, batch_size)
    for chunk in chunks:
        yield [ins.identifier for ins in chunk]


class MockProbaModel(Generic[KT, LT]):
    labelset: Sequence[LT]
    labels: LabelProvider[KT, LT]

    def __init__(self, labelprovider: LabelProvider[KT, LT]):
        self.labels = labelprovider
        self.labelset = sorted(list(labelprovider.labelset))  # type: ignore

    def __call__(self, keys: Sequence[KT]) -> npt.NDArray[np.float64]:
        result = np.zeros((len(keys), len(self.labelset)))
        for i, key in enumerate(keys):
            given_labels = self.labels[key]
            for j, lbl in enumerate(self.labelset):
                result[i, j] = np.float64(lbl in given_labels)
        return result

    @property
    def encoder(self) -> LabelEncoder:
        inverted_mapping = invert_mapping(seq_or_map_to_map(self.labelset))
        encoder = MultilabelDictionaryEncoder(inverted_mapping)
        return encoder
    



class MockClassifierWrapper(
    AbstractClassifier[IT, KT, DT, VT, RT, LT, LMT, PMT],
    Generic[IT, KT, DT, VT, RT, LT, LVT, LMT, PMT],
):
    proba_vec_function: Callable[[Sequence[KT]], PMT]

    def __init__(
        self,
        proba_function: Callable[[Sequence[KT]], PMT],
        threshold_func: Callable[[PMT], LMT],
        encoder: LabelEncoder[LT, LVT, LMT, PMT],
    ) -> None:
        self.proba_vec_function = proba_function
        self.threshold_function = threshold_func
        self.encoder = encoder

    @property
    def fitted(self) -> bool:
        return True

    def fit_instances(
        self,
        instances: Iterable[Instance[KT, DT, VT, RT]],
        labels: Iterable[Iterable[LT]],
    ) -> None:
        pass

    def fit_provider(
        self,
        provider: InstanceProvider[IT, KT, DT, VT, RT],
        labels: LabelProvider[KT, LT],
        batch_size: int = 200,
    ) -> None:
        pass

    def _get_probas(self, keys: Sequence[KT]) -> Tuple[Sequence[KT], PMT]:
        """Calculate the probability matrix for the current (key, data) tuples

        Parameters
        ----------
        tuples : Sequence[Tuple[KT, DT]]
            The tuples that we want the predictions from

        Returns
        -------
        Tuple[Sequence[KT], npt.NDArray[Any]]
            A list of keys and the probability predictions belonging to it
        """
        y_pred = self.proba_vec_function(keys)
        return keys, y_pred

    def _proba_iterator(
        self, datas: Sequence[KT]
    ) -> Sequence[Tuple[KT, FrozenSet[Tuple[LT, float]]]]:
        keys, y_pred = self._get_probas(datas)
        labels = self.encoder.decode_proba_matrix(y_pred)
        return list(zip(keys, labels))

    def _pred_iterator(self, keys: Sequence[KT]) -> Sequence[Tuple[KT, FrozenSet[LT]]]:
        return self._decode_proba_matrix_pred(*self._get_probas(keys))

    def predict_proba_instances(
        self,
        instances: Iterable[Instance[KT, DT, VT, RT]],
        batch_size: int = 200,
    ) -> Sequence[Tuple[KT, FrozenSet[Tuple[LT, float]]]]:
        decoded_probas = map(self._proba_iterator, key_chunker(instances, 200))
        chained = list(itertools.chain.from_iterable(decoded_probas))
        return chained

    def _decode_proba_matrix(
        self, keys: Sequence[KT], y_matrix: PMT
    ) -> Sequence[Tuple[KT, FrozenSet[Tuple[LT, float]]]]:
        y_labels = self.encoder.decode_proba_matrix(y_matrix)
        zipped = list(zip(keys, y_labels))
        return zipped

    def _decode_proba_matrix_pred(
        self, keys: Sequence[KT], y_matrix: PMT
    ) -> Sequence[Tuple[KT, FrozenSet[LT]]]:
        thresholded = self.threshold_function(y_matrix)
        y_labels = self.encoder.decode_matrix(thresholded)
        zipped = list(zip(keys, y_labels))
        return zipped

    def predict_instances(
        self,
        instances: Iterable[Instance[KT, DT, VT, RT]],
        batch_size: int = 200,
    ) -> Sequence[Tuple[KT, FrozenSet[LT]]]:
        decoded_probas = map(self._pred_iterator, key_chunker(instances, 200))
        chained = list(itertools.chain.from_iterable(decoded_probas))
        return chained

    def predict_provider(
        self,
        provider: InstanceProvider[IT, KT, DT, VT, RT],
        batch_size: int = 200,
    ) -> Sequence[Tuple[KT, FrozenSet[LT]]]:
        keys = divide_iterable_in_lists(provider.key_list, batch_size)
        decoded_probas = map(self._pred_iterator, keys)
        chained = list(itertools.chain.from_iterable(decoded_probas))
        return chained

    def predict_proba_provider_raw(
        self,
        provider: InstanceProvider[IT, KT, DT, Any, Any],
        batch_size: int = 200,
    ) -> Iterator[Tuple[Sequence[KT], PMT]]:
        keys = divide_iterable_in_lists(provider.key_list, batch_size)
        total_it = math.ceil(len(provider) / batch_size)
        preds = map(self._get_probas, tqdm(keys, total=total_it, leave=False))
        yield from preds

    def predict_proba_instances_raw(
        self,
        instances: Iterable[Instance[KT, DT, VT, RT]],
        batch_size: int = 200,
    ) -> Iterator[Tuple[Sequence[KT], PMT]]:
        keys = key_chunker(instances, batch_size)
        preds = map(self._get_probas, tqdm(keys, leave=False))
        yield from preds

    def predict_proba_provider(
        self,
        provider: InstanceProvider[IT, KT, DT, VT, Any],
        batch_size: int = 200,
    ) -> Sequence[Tuple[KT, FrozenSet[Tuple[LT, float]]]]:
        preds = self.predict_proba_provider_raw(provider, batch_size)
        decoded_probas = itertools.starmap(self._decode_proba_matrix, preds)
        chained = list(itertools.chain.from_iterable(decoded_probas))
        return chained

    def get_label_column_index(self, label: LT) -> int:
        return self.encoder.get_label_column_index(label)

    def set_target_labels(self, labels: Iterable[LT]) -> None:
        pass

    @classmethod
    def from_provider(cls, provider: LabelProvider[KT, LT]) -> Self:
        mock_model = MockProbaModel(provider)
        threshold = numpy_ova_threshold(0.5)
        wrapped = cls(mock_model, threshold, mock_model.encoder)  # type: ignore
        return wrapped

    @classmethod
    def from_env(cls, env: AbstractEnvironment[IT, KT, DT, VT, RT, LT]) -> Self:
        return cls.from_provider(env.truth)


