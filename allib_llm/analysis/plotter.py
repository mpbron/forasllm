from __future__ import annotations

import typing as ty
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from os import PathLike
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import instancelib as il
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from allib.activelearning.base import ActiveLearner
from allib.activelearning.ensembles import AbstractEnsemble
from allib.analysis.experiments import ExperimentIterator
from allib.analysis.plotter import ExperimentPlotter
from allib.analysis.statistics import ModelNamer, TarDatasetStats
from allib.analysis.tarplotter import escape, smooth_similar, smooth_similar3
from allib.estimation.base import AbstractEstimator, Estimate
from allib.typehints import KT, LT
from instancelib.utils.func import flatten_dicts
from typing_extensions import Self

from ..utils.instances import LabelTransformer
from .performance import process_performance


@dataclass
class CriteriaDatasetStats(TarDatasetStats):
    @classmethod
    def from_learner(
        cls,
        learner: ActiveLearner[Any, Any, Any, Any, Any, LT],
        label_transformer: LabelTransformer[KT, LT],
        pos_label: LT,
        neg_label: LT,
    ) -> Self:
        binary_truth = label_transformer(learner.env.truth, pos_label, neg_label)
        pos_count = len(
            learner.env.get_subset_by_labels(
                learner.env.dataset, pos_label, labelprovider=binary_truth
            )
        )
        neg_count = len(
            learner.env.get_subset_by_labels(
                learner.env.dataset, neg_label, labelprovider=binary_truth
            )
        )
        size = len(learner.env.dataset)
        prevalence = pos_count / size
        return cls(pos_count, neg_count, size, prevalence)


@dataclass
class LLMRecallStats:
    name: str
    wss: float
    recall: float
    proportional_effort: float
    pos_docs_found: int
    neg_docs_found: int
    effort: int
    loss_er: float
    child_statistics: Sequence[Self]

    @classmethod
    def from_learner(
        cls,
        learner: ActiveLearner[Any, Any, Any, Any, Any, LT],
        label_transformer: LabelTransformer[KT, LT],
        pos_label: LT,
        neg_label: LT,
        default: Optional[ModelNamer] = None,
    ) -> Self:
        namer = ModelNamer() if default is None else default
        str_id = namer(learner.name)
        perf = process_performance(learner, pos_label, neg_label, label_transformer)
        pos_docs = len(perf.true_positives)
        neg_docs = len(perf.false_positives)
        effort = len(learner.env.labeled)
        prop_effort = effort / len(learner.env.dataset)
        if isinstance(learner, AbstractEnsemble) and not hasattr(
            learner, "_disable_substats"
        ):
            subresults: Sequence[Self] = tuple(
                [
                    cls.from_learner(learner, label_transformer, pos_label, neg_label, namer)
                    for learner in learner.learners
                ]
            )
        else:
            subresults = tuple()
        return cls(
            str_id,
            perf.wss,
            perf.recall,
            prop_effort,
            pos_docs,
            neg_docs,
            effort,
            perf.loss_er,
            subresults
        )
    
    def flatten(self) -> Mapping[str, Self]:
        root = {self.name: self}
        children = flatten_dicts(*[stat.flatten() for stat in self.child_statistics])
        return {**root, **children}

    @classmethod
    def transpose_dict(cls,
        recall_stats: Mapping[int, Self]
    ) -> Mapping[str, Mapping[int, Self]]:
        flattened = {key: stat.flatten() for key, stat in recall_stats.items()}
        if flattened:
            learner_names = list(next(iter(flattened.values())).keys())
            ret = {
                learner: {t: substat[learner] for t, substat in flattened.items()}
                for learner in learner_names
            }
            return ret
        return dict()


class CriteriaTarPlotter(ExperimentPlotter[LT], Generic[LT]):
    pos_label: LT
    neg_label: LT
    label_transformer: LabelTransformer

    dataset_name: str

    dataset_stats: ty.OrderedDict[int, TarDatasetStats]
    recall_stats: ty.OrderedDict[int, LLMRecallStats]
    estimates: ty.OrderedDict[int, Mapping[str, Estimate]]
    stop_results: ty.OrderedDict[int, Mapping[str, bool]]

    def __init__(self, pos_label: LT, neg_label: LT, label_transformer, dataset_name: str = "") -> None:
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.label_transformer = label_transformer
        self.dataset_name = dataset_name

        self.dataset_stats = OrderedDict()
        self.recall_stats = OrderedDict()
        self.estimates = OrderedDict()
        self.stop_results = OrderedDict()
        self.it = 0
        self.it_axis: List[int] = list()

    def update(
        self,
        exp_iterator: ExperimentIterator[Any, Any, Any, Any, Any, LT],
        stop_result: Mapping[str, bool],
    ) -> None:
        learner = exp_iterator.learner
        self.it = exp_iterator.it
        self.it_axis.append(self.it)
        self.recall_stats[self.it] = LLMRecallStats.from_learner(
            learner, self.label_transformer, self.pos_label, self.neg_label
        )
        self.dataset_stats[self.it] = CriteriaDatasetStats.from_learner(
            learner, self.label_transformer, self.pos_label, self.neg_label
        )
        self.estimates[self.it] = {
            name: estimate for name, estimate in exp_iterator.recall_estimate.items()
        }
        self.stop_results[self.it] = stop_result

    @property
    def estimator_names(self) -> FrozenSet[str]:
        if self.estimates:
            return frozenset(self.estimates[self.it].keys())
        return frozenset()

    @property
    def criterion_names(self) -> FrozenSet[str]:
        if self.stop_results:
            return frozenset(self.stop_results[self.it].keys())
        return frozenset()

    def _effort_axis(self) -> npt.NDArray[Any]:
        effort_axis = np.array([self.recall_stats[it].effort for it in self.it_axis])
        return effort_axis

    def exp_random_recall(self, it: int) -> float:
        effort = self.recall_stats[it].effort
        dataset_size = self.dataset_stats[it].size
        true_pos = self.dataset_stats[it].pos_count
        expected = effort / dataset_size * true_pos
        return expected

    def print_last_stats(self) -> None:
        estimate = self.estimates[self.it]
        recall = self.recall_stats[self.it]
        print(estimate)
        print(recall.pos_docs_found)

    def plot_recall_statistic(
        self, stats: Mapping[int, LLMRecallStats], key: str, label: str
    ) -> None:
        effort_axis = self._effort_axis()
        curve = np.array([stats[it].__dict__[key] for it in self.it_axis])
        if label[0] == "_":
            plt.plot(effort_axis, curve, label=label, alpha=0.5, color="gray")
        else:
            plt.plot(effort_axis, curve, label=label)

    def _plot_estimator(
        self,
        key: str,
        color: str = "gray",
        alpha: float = 0.4,
        latex: bool = False,
        rename_dict: Mapping[str, str] = dict(),
    ) -> None:
        effort_axis = self._effort_axis()
        points = np.array([self.estimates[it][key].point for it in self.it_axis])
        lows = np.array([self.estimates[it][key].lower_bound for it in self.it_axis])
        uppers = np.array([self.estimates[it][key].upper_bound for it in self.it_axis])
        xs, ys = effort_axis, points  # smooth_similar(effort_axis, points)
        xrs, ls, us = (
            effort_axis,
            lows,
            uppers,
        )  # smooth_similar3(effort_axis, lows, uppers)
        estimator_name = key if key not in rename_dict else rename_dict[key]
        plt.plot(
            xs,
            ys,
            linestyle="-.",
            label=escape(f"Estimate by {estimator_name}", latex),
            color=color,
        )
        plt.fill_between(xrs, ls, us, color=color, alpha=alpha)  # type: ignore

    def _plot_stop_criteria(
        self,
        included_criteria: Optional[Sequence[str]],
        show_stats=True,
        show_wss=False,
        show_recall=False,
        latex=False,
        rename_dict: Mapping[str, str] = dict(),
    ) -> None:
        results: Sequence[Tuple[int, float, float, str, str]] = list()
        if included_criteria is None:
            included_criteria = list(self.criterion_names)
        for i, crit_name in enumerate(included_criteria):
            color = f"C{i}"
            for it in self.it_axis:
                frame = self.stop_results[it]
                if frame[crit_name]:
                    wss = self.recall_stats[it].wss
                    recall = self.recall_stats[it].recall
                    results.append((it, recall, wss, crit_name, color))
                    break
        results_sorted = sorted(results)
        for it, recall, wss, crit_name, color in results_sorted:
            exp_found = self.exp_random_recall(it)
            act_found = self.recall_stats[it].pos_docs_found
            criterion_name = (
                crit_name if crit_name not in rename_dict else rename_dict[crit_name]
            )
            nicer_name = criterion_name.replace("_", " ")
            if show_stats:
                legend = (
                    f"{nicer_name} \n WSS: {(wss*100):.1f} Recall: {(recall*100):.1f} %"
                )
            elif show_recall:
                legend = f"{nicer_name} ({(recall*100):.1f} %)"
            elif show_wss:
                legend = f"{nicer_name} - {(wss*100):.1f} %"
            else:
                legend = f"{nicer_name}"
            plt.vlines(
                x=self.recall_stats[it].effort,
                ymin=exp_found,
                ymax=act_found,
                linestyles="dashed",
                color=color,
                label=escape(legend, latex),
            )

    def _graph_setup(self, simulation=True, latex=False) -> None:
        if latex:
            plt.style.use(["science", "nature"])
        true_pos = self.dataset_stats[self.it].pos_count
        dataset_size = self.dataset_stats[self.it].size

        plt.xlabel(f"number of read documents")
        plt.ylabel("number of retrieved relevant documents")
        if simulation:
            plt.title(
                f"Run on a dataset with {int(true_pos)} inclusions out of {int(dataset_size)}"
            )
        else:
            plt.title(f"Run on a dataset of {int(dataset_size)}")

    def _plot_static_data(self, recall_target: float, latex=False) -> None:
        # Static data
        true_pos = self.dataset_stats[self.it].pos_count
        dataset_size = self.dataset_stats[self.it].size

        pos_target = int(np.ceil(recall_target * true_pos))
        effort_axis = self._effort_axis()

        plt.axhline(
            y=true_pos, linestyle=":", label=escape(f"100 % recall ({true_pos})", latex)
        )
        plt.axhline(
            y=pos_target,
            linestyle=":",
            label=escape(f"{int(recall_target * 100)} % recall ({pos_target})", latex),
        )
        plt.plot(
            effort_axis,
            (effort_axis / dataset_size) * true_pos,
            ":",
            label=f"Exp. found at random",
        )

    def _plot_recall_stats(
        self,
        included: Optional[Sequence[str]] = list(),
        short_names=False,
        latex=False,
        rename_dict: Mapping[str, str] = dict(),
        show_only_legend: Sequence[str] = list(),
    ) -> None:
        # Gather and reorganize recall data
        recall_stats = LLMRecallStats.transpose_dict(self.recall_stats)
        # Plot pos docs docs found
        for name, stats in recall_stats.items():
            if short_names:
                try:
                    pname = name.split("-")[0].rstrip().lstrip()
                except:
                    pname = name
            else:
                pname = name
            if included is None or name in included:
                model_name = pname if pname not in rename_dict else rename_dict[pname]
                if show_only_legend and name not in show_only_legend:
                    self.plot_recall_statistic(
                        stats, "pos_docs_found", f"_{model_name}"
                    )
                else:
                    self.plot_recall_statistic(
                        stats, "pos_docs_found", escape(f"# by {model_name}", latex)
                    )

    def _plot_estimators(
        self,
        included_estimators: Optional[Sequence[str]] = None,
        latex=False,
        rename_dict: Mapping[str, str] = dict(),
    ) -> None:
        if included_estimators is None:
            included_estimators = list(self.estimator_names)
        # Plotting estimations
        for i, estimator in enumerate(included_estimators):
            self._plot_estimator(estimator, color=f"C{i*2}", rename_dict=rename_dict)

    def _set_axes(
        self, x_lim: Optional[float] = None, y_lim: Optional[float] = None
    ) -> None:
        # Setting axis limitations
        true_pos = self.dataset_stats[self.it].pos_count
        if x_lim is not None:
            plt.xlim(0, x_lim)
        if y_lim is not None:
            plt.ylim(0, y_lim)
        else:
            plt.ylim(0, 1.4 * true_pos)

    def _plot_legend(self, latex=False) -> None:
        if latex:
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        else:
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize="xx-small")

    def show(
        self,
        x_lim: Optional[float] = None,
        y_lim: Optional[float] = None,
        recall_target: float = 0.95,
        included_estimators: Optional[Sequence[str]] = None,
        included_models: Optional[Sequence[str]] = None,
        included_stopcriteria: Optional[Sequence[str]] = None,
        filename: "Optional[PathLike[str]]" = None,
        latex: bool = False,
        show_stats=True,
        short_names=False,
        rename_models=dict(),
        rename_estimators=dict(),
        rename_criteria=dict(),
        show_recall=False,
        show_wss=True,
        show_only_models=list(),
    ) -> None:
        self._graph_setup(latex=latex)
        self._plot_static_data(recall_target, latex=latex)
        self._plot_recall_stats(
            included_models,
            latex=latex,
            short_names=short_names,
            show_only_legend=show_only_models,
        )
        self._plot_estimators(
            included_estimators, latex=latex, rename_dict=rename_estimators
        )
        self._plot_stop_criteria(
            included_stopcriteria,
            latex=latex,
            show_stats=show_stats,
            show_recall=show_recall,
            show_wss=show_wss,
            rename_dict=rename_criteria,
        )
        self._set_axes(x_lim, y_lim)
        self._plot_legend(latex=latex)
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    def user_show(
        self,
        x_lim: Optional[float] = None,
        y_lim: Optional[float] = None,
        included_estimators: Optional[Sequence[str]] = None,
        included_models: Optional[Sequence[str]] = None,
        included_stopcriteria: Optional[Sequence[str]] = None,
        filename: "Optional[PathLike[str]]" = None,
        latex: bool = False,
        short_names=False,
    ):
        self._graph_setup(simulation=False, latex=latex)
        self._plot_recall_stats(included_models, latex=latex, short_names=short_names)
        self._plot_estimators(included_estimators, latex=latex)
        self._plot_stop_criteria(included_stopcriteria, latex=latex, show_stats=False)
        self._set_axes(x_lim, y_lim)
        self._plot_legend(latex=latex)
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    def wss_at_target(self, target: float) -> float:
        for it in self.it_axis:
            frame = self.recall_stats[it]
            if frame.recall >= target:
                return frame.wss
        return float("nan")

    def recall_at_stop(self, stop_criterion: str) -> float:
        stop_it = self._it_at_stop(stop_criterion)
        if stop_it is not None:
            return self.recall_stats[stop_it].recall
        return float("nan")

    def _it_at_stop(self, stop_criterion: str) -> Optional[int]:
        for it in self.it_axis:
            frame = self.stop_results[it]
            if frame[stop_criterion]:
                return it
        return None

    def wss_at_stop(self, stop_criterion: str) -> float:
        stop_it = self._it_at_stop(stop_criterion)
        if stop_it is not None:
            return self.recall_stats[stop_it].wss
        return float("nan")

    def relative_error(self, stop_criterion: str, recall_target: float) -> float:
        recall_at_stop = self.recall_at_stop(stop_criterion)
        relative_error = abs(recall_at_stop - recall_target) / recall_target
        return relative_error

    def loss_er_at_stop(self, stop_criterion: str) -> float:
        stop_it = self._it_at_stop(stop_criterion)
        if stop_it is not None:
            return self.recall_stats[stop_it].loss_er
        return float("nan")

    def effort_at_stop(self, stop_criterion: str) -> int:
        stop_it = self._it_at_stop(stop_criterion)
        if stop_it is not None:
            return self.recall_stats[stop_it].effort
        return self.recall_stats[self.it_axis[-1]].effort

    def proportional_effort_at_stop(self, stop_criterion: str) -> float:
        stop_it = self._it_at_stop(stop_criterion)
        if stop_it is not None:
            return self.recall_stats[stop_it].proportional_effort
        return 1.0