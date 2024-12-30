# %%
from dataclasses import dataclass
from typing import Any, Callable, Generic, Mapping, Sequence, Tuple

import instancelib as il
from instancelib.utils.func import value_map
from allib.environment.base import AbstractEnvironment
from allib.activelearning.base import ActiveLearner
from allib.machinelearning.taroptimized import ALSklearn
from allib.typehints.typevars import DT, IT, KT, LT, RT, VT
from cleanlab.classification import CleanLearning
from instancelib.analysis.base import BinaryModelMetrics, compare_models
from sklearn.linear_model import LogisticRegression

from allib_llm.machinelearning.mockclassifier import MockClassifierWrapper
from allib.configurations.tarbaselines import autotar
from allib.analysis.experiments import ExperimentIterator
from allib.stopcriterion.heuristic import AprioriRecallTarget
from allib.analysis.tarplotter import TarExperimentPlotter
from allib.analysis.simulation import TarSimulator
from allib.environment.base import AbstractEnvironment
from allib.stopcriterion.base import AbstractStopCriterion
from allib.feature_extraction.abstracts import CombinedVectorizer
from allib.stopcriterion.others import StopAfterKNegative
from allib.machinelearning.sparse import SparseVectorStorage
from allib.analysis.analysis import BinaryPerformance
from allib.activelearning.autotar import pseudo_from_metadata
from allib_llm.simulation.labelmock import binary_error_rate, mock_errors
from ..machinelearning.labelcorrection import (
    noisy_label_correction,
    find_likely_relevant,
)
from allib.analysis.analysis import process_performance
from typing_extensions import Self


@dataclass(frozen=True)
class Workload(Generic[KT]):
    docs: Sequence[KT]
    pos_count: int
    neg_count: int
    length: int

    @classmethod
    def from_provider(
        cls,
        provider: il.InstanceProvider[IT, KT, DT, VT, RT],
        env: AbstractEnvironment[IT, KT, DT, VT, RT, LT],
        ground_truth: il.LabelProvider[KT, LT],
        pos_label: LT,
        neg_label: LT,
    ):
        poss = env.get_subset_by_labels(provider, pos_label, labelprovider=ground_truth)
        negs = env.get_subset_by_labels(provider, neg_label, labelprovider=ground_truth)
        return cls(tuple(provider.key_list), len(poss), len(negs), len(provider))

    @classmethod
    def from_metrics(cls, metrics: BinaryModelMetrics[KT, Any]) -> Self:
        docs = (*metrics.true_positives, *metrics.false_positives)
        return cls(
            docs, len(metrics.true_positives), len(metrics.false_positives), len(docs)
        )


@dataclass(frozen=True)
class ModelCondition:
    name: str
    pos_recall: float
    neg_recall: float

    @classmethod
    def from_results(cls, name: str, results: BinaryModelMetrics[Any, Any]) -> Self:
        pos_recall = results.recall
        neg_recall = len(results.true_negatives) / (
            len(results.true_negatives) + len(results.false_positives)
        )
        return cls(name, pos_recall, neg_recall)


@dataclass(frozen=True)
class SimulationResult(Generic[KT, LT]):
    ds_name: str
    model_condition: ModelCondition
    pos_count: int
    neg_count: int
    size: int
    llm_workload: Workload[KT]
    ec_workload: Workload[KT]
    lp_workload: Workload[KT]
    al_workloads: Mapping[str, Workload[KT]]
    llm_metrics: BinaryModelMetrics[KT, LT]
    ec_metrics: BinaryModelMetrics[KT, LT]
    al_metrics: Mapping[str, BinaryModelMetrics[KT, LT]]


def al_simulation(
    env: AbstractEnvironment[IT, KT, DT, VT, Any, LT],
    al_builder: Callable[..., ActiveLearner[IT, KT, DT, VT, Any, LT]],
    stop_criterion_builders: Mapping[str, Callable[..., AbstractStopCriterion[LT]]],
    pos_label: LT,
    neg_label: LT,
) -> Tuple[BinaryModelMetrics[KT, LT], ActiveLearner[IT, KT, DT, VT, Any, LT], TarExperimentPlotter]:
    env = env.from_environment(env, shared_labels=False)
    al = al_builder(env, pos_label=pos_label, neg_label=neg_label)
    criteria = value_map(lambda x: x(pos_label, neg_label), stop_criterion_builders)
    exp = ExperimentIterator(
        al, pos_label, neg_label, criteria, dict()
    )
    plotter = TarExperimentPlotter(pos_label, neg_label)
    simulator = TarSimulator(exp, plotter, stop_when_satisfied=True)
    simulator.simulate()
    performance = process_performance(al, pos_label)
    return performance, al, plotter


def alsklearn_builder(
    clf: Any,
    vectorizer: il.BaseVectorizer[IT],
    vectorstorage_builder: Callable[[], SparseVectorStorage],
) -> Callable[
    [AbstractEnvironment[IT, KT, DT, VT, Any, LT]], ALSklearn[IT, KT, DT, VT, LT, Any]
]:
    def builder(
        env: AbstractEnvironment[IT, KT, DT, VT, Any, LT]
    ) -> ALSklearn[IT, KT, DT, VT, LT, Any]:
        return ALSklearn.build(
            clf, env, vectorizer=vectorizer, vectorstorage_builder=vectorstorage_builder
        )

    return builder


def simulate_llm_correction(
    ds_name: str,
    env: AbstractEnvironment[IT, KT, DT, VT, Any, LT],
    model_condition: ModelCondition,
    llm_classifier: il.AbstractClassifier,
    classifier_builder: Callable[
        [AbstractEnvironment[IT, KT, DT, VT, Any, LT]],
        ALSklearn[IT, KT, DT, VT, LT, Any],
    ],
    al_builders: Mapping[str, Callable[..., ActiveLearner[IT, KT, DT, VT, Any, LT]]],
    stop_criterion_builder: Mapping[str, Callable[..., AbstractStopCriterion[LT]]],
    pos_label: LT,
    neg_label: LT,
    scale_factor: float = 2,
) -> SimulationResult[KT, LT]:
    lr = LogisticRegression()
    cl = CleanLearning(lr, cv_n_folds=5)
    insclf = classifier_builder(env)
    llm_labels = il.MemoryLabelProvider.from_tuples(llm_classifier.predict(env.dataset))
    llm_corrected_workload = Workload.from_provider(
        noisy_label_correction(
            env, llm_labels, pos_label, neg_label, cl, insclf, scale_factor
        ),
        env,
        env.truth,
        pos_label,
        neg_label,
    )
    llm_likely_relevant_workload = Workload.from_provider(
        find_likely_relevant(env, llm_labels, pos_label, neg_label, cl, insclf),
        env,
        env.truth,
        pos_label,
        neg_label,
    )
    al_results = value_map(
        lambda al_builder: al_simulation(
            env, al_builder, stop_criterion_builder, pos_label, neg_label
        ),
        al_builders,
    )
    al_results_stats = value_map(lambda x: x[0], al_results)
    models = {"LLM": llm_classifier, "EC": insclf}
    model_results = value_map(
        BinaryPerformance[KT, LT].from_bm_metrics,
        compare_models(env.dataset, env.truth, pos_label, models),
    )
    llm_orig_workload = Workload.from_metrics(model_results["LLM"])
    al_workloads = value_map(Workload.from_metrics, al_results)  # type: ignore
    result = SimulationResult(
        ds_name,
        model_condition,
        env.truth.document_count(pos_label),
        env.truth.document_count(neg_label),
        len(env.dataset),
        llm_orig_workload,
        llm_corrected_workload,
        llm_likely_relevant_workload,
        al_workloads,
        model_results["LLM"],
        model_results["EC"],
        al_results_stats,
    )
    return result
