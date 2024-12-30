import logging
from os import PathLike
from typing import Any, Callable, Generic, Optional, Union

import instancelib as il
import numpy.typing as npt
from allib.balancing.base import BaseBalancer, IdentityBalancer
from allib.machinelearning.taroptimized import ALSklearn
from allib.typehints.typevars import IT
from instancelib.instances.vectorstorage import VectorStorage
from instancelib.labels.encoder import DictionaryEncoder
from instancelib.typehints.typevars import DT, KT, LT, VT, DType
from instancelib.utils.saveablemodel import SaveableInnerModel
from sklearn.base import ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import LabelEncoder as SKLabelEncoder
from typing_extensions import Self

LOGGER = logging.getLogger(__name__)

class CriteriaClassifier(ALSklearn[IT, KT, DT, VT, LT, DType], Generic[IT, KT, DT, VT, LT, DType]):

    @SaveableInnerModel.load_model_fallback
    def _fit(self, x_data: npt.NDArray[Any], y_data: npt.NDArray[Any]):
        assert x_data.shape[0] == y_data.shape[0]
        if len(y_data.shape) == 2 and y_data.shape[1] == 1:
            y_data = y_data.ravel()
        self.innermodel.fit(x_data, y_data)  # type: ignore
        LOGGER.info("[%s] Fitted the model", self.name)
        self._fitted = True

    @classmethod
    def build_binary(
        cls,
        estimator: Union[ClassifierMixin, Pipeline],
        env: il.Environment[IT, KT, DT, VT, Any, LT],
        vectorizer: il.BaseVectorizer[IT],
        vectorstorage_builder: Callable[
            [], VectorStorage[KT, npt.NDArray[DType], npt.NDArray[DType]]
        ],
        pos_label: LT,
        neg_label: LT,
        balancer: BaseBalancer = IdentityBalancer(),
        chunk_size: int = 2000,
        storage_location: "Optional[PathLike[str]]" = None,
        filename: "Optional[PathLike[str]]" = None,
    ) -> Self:
        """Construct a Sklearn model from an :class:`~instancelib.Environment`.
        The estimator is a classifier for a binary or multiclass classification problem.

        Parameters
        ----------
        estimator : ClassifierMixin
            The Sklearn Classifier (e.g., :class:`sklearn.naive_bayes.MultinomialNB`)
        env : Environment[IT, KT, Any, npt.NDArray[Any], Any, LT]
            The environment that will be used to gather the labels from
        storage_location : Optional[PathLike[str]], optional
            If you want to save the model, you can specify the storage folder, by default None
        filename : Optional[PathLike[str]], optional
            If the model has a specific filename, you can specify it here, by default None

        Returns
        -------
        SkLearnClassifier[IT, KT, DT, LT]
            The model
        """
        il_encoder = DictionaryEncoder({pos_label: 1, neg_label: 0})
        vectorstorage = cls.vectorize(
            env, vectorizer, vectorstorage_builder(), chunk_size
        )
        return cls(
            estimator,
            il_encoder,
            vectorizer,
            vectorstorage,
            balancer,
            storage_location,
            filename,
        )
    
    @classmethod
    def build_from_labels(
        cls,
        estimator: Union[ClassifierMixin, Pipeline],
        env: il.Environment[IT, KT, DT, VT, Any, LT],
        labels: il.LabelProvider[KT, LT],
        vectorizer: il.BaseVectorizer[IT],
        vectorstorage_builder: Callable[
            [], VectorStorage[KT, npt.NDArray[DType], npt.NDArray[DType]]
        ],        
        balancer: BaseBalancer = IdentityBalancer(),
        chunk_size: int = 2000,
        storage_location: "Optional[PathLike[str]]" = None,
        filename: "Optional[PathLike[str]]" = None,
    ) -> Self:
        """Construct a Sklearn model from an :class:`~instancelib.Environment`.
        The estimator is a classifier for a binary or multiclass classification problem.

        Parameters
        ----------
        estimator : ClassifierMixin
            The Sklearn Classifier (e.g., :class:`sklearn.naive_bayes.MultinomialNB`)
        env : Environment[IT, KT, Any, npt.NDArray[Any], Any, LT]
            The environment that will be used to gather the labels from
        storage_location : Optional[PathLike[str]], optional
            If you want to save the model, you can specify the storage folder, by default None
        filename : Optional[PathLike[str]], optional
            If the model has a specific filename, you can specify it here, by default None

        Returns
        -------
        SkLearnClassifier[IT, KT, DT, LT]
            The model
        """
        il_encoder = DictionaryEncoder.from_list(list(labels.labelset))
        vectorstorage = cls.vectorize(
            env, vectorizer, vectorstorage_builder(), chunk_size
        )
        return cls(
            estimator,
            il_encoder,
            vectorizer,
            vectorstorage,
            balancer,
            storage_location,
            filename,
        )
