# %%
import pickle
from pathlib import Path
from typing import Sequence

import instancelib as il
import pandas as pd
from allib.activelearning.random import RandomSampling
from allib.analysis.experiments import ExperimentIterator
from allib.analysis.simulation import TarSimulator
from allib.feature_extraction.abstracts import CombinedVectorizer
from allib.machinelearning.sparse import SparseVectorStorage
from allib.stopcriterion.heuristic import DocCountStopCritertion
from environs import Env
from instancelib.typehints.typevars import KT, LT
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from allib_llm.activelearning.criteriaseparate import CriteriaWSA, MinSample, SkipNegWSA
from allib_llm.activelearning.errorcorrector import TwoPhase
from allib_llm.activelearning.static import FixedOrdering
from allib_llm.activelearning.wsa import (
    Acceptance,
    AutoTARReduced,
    AutoTarWSA,
    LLMPreferred,
)
from allib_llm.analysis.performance import process_performance
from allib_llm.analysis.plotter import CriteriaTarPlotter
from allib_llm.datasets.foras import (
    answers_to_provider,
    foras_to_env,
    symbol_label_viewer,
)
from allib_llm.machinelearning.criteria_llm import CriteriaLLMClassifier
from allib_llm.machinelearning.criteriaprobafunc import MockProbaClassifier
from allib_llm.machinelearning.labelreduction import CriteriaClassifier
from allib_llm.machinelearning.langchain import QAResult
from allib_llm.machinelearning.mockclassifier import MockClassifierWrapper
from allib_llm.stopcriterion.wrapper import (
    AprioriCriteriaRecallTarget,
    BudgetCriteriaStoppingRule,
    SameStateCount,
)
from allib_llm.utils.instances import (
    BinarySubSetter,
    CriteriaToBinary,
    DefaultSubSetter,
    LabelTransformer,
    SimpleProbaProvider,
    SubLabelProviderSplitter,
    conjunction_judgement_weakly,
    get_subset,
)

# %%
POS = "Relevant"
NEG = "Irrelevant"
BPATH = Path("./data/")
DS_PATH = BPATH / "sourcedata" / "Motherfile_270224_V3.xlsx"
DS_NAME = DS_PATH.stem
ENV = Env()
ENV.read_env(".env")
ds_env = foras_to_env(DS_PATH)

# %%

basepath = BPATH / "llm_results" / "GPTModel.GPT35" / "questions"
df_path = basepath / DS_NAME / "results.pkl"
obj_path = basepath / DS_NAME / "results_obj.pkl"

# %%
with obj_path.open("rb") as fh:
    objs: Sequence[QAResult] = pickle.load(fh)
qa_dict = {o.key: o for o in objs}
df = pd.read_pickle(df_path)
mf = pd.read_excel(DS_PATH)
cleaned = ds_env.to_pandas(
    ds_env.dataset, labels=ds_env.truth, label_viewer=symbol_label_viewer()
)

# %%
pred_provider = answers_to_provider(objs)

# %%
lbltrans = LabelTransformer(conjunction_judgement_weakly)
full_clf = CriteriaLLMClassifier.from_answers(objs)
proba_clf = MockProbaClassifier.from_provider(pred_provider, POS, NEG)
binary_truth = lbltrans(ds_env.truth, POS, NEG)
# %%
llm_tar = FixedOrdering.builder(full_clf)


def clf_builder(env):
    lr = LogisticRegression(solver="lbfgs", C=1.0, max_iter=10000)
    vect = CombinedVectorizer(TfidfVectorizer(max_features=3000))
    return CriteriaClassifier.build_binary(lr, env, vect, SparseVectorStorage, POS, NEG)


def clf_builder_lbl(env: il.Environment, lbl_provider: il.LabelProvider[KT, LT]):
    lr = LogisticRegression(solver="lbfgs", C=1.0, max_iter=10000)
    vect = CombinedVectorizer(TfidfVectorizer(max_features=3000))
    return CriteriaClassifier.build_from_labels(
        lr, env, lbl_provider, vect, SparseVectorStorage
    )


def clf_builder_lbl_cv(env: il.Environment, lbl_provider: il.LabelProvider[KT, LT]):
    lr = LogisticRegressionCV(solver="lbfgs", max_iter=10000)
    vect = CombinedVectorizer(TfidfVectorizer(max_features=3000))
    return CriteriaClassifier.build_from_labels(
        lr, env, lbl_provider, vect, SparseVectorStorage
    )


def acc_builder(env):
    lr = LogisticRegressionCV(solver="lbfgs", max_iter=10000)
    vect = CombinedVectorizer(TfidfVectorizer(max_features=3000))
    return CriteriaClassifier.build_binary(
        lr,
        env,
        vect,
        SparseVectorStorage,
        Acceptance.ACCEPT,
        Acceptance.REJECT,
    )


lbltrans = LabelTransformer(conjunction_judgement_weakly)
llm_prefer_builder = LLMPreferred.builder_alt(clf_builder, full_clf, lbltrans, 100, 10)
llm_wsa_tar_builder = AutoTarWSA.builder_alt(
    clf_builder, full_clf, acc_builder, lbltrans, 100, 10
)
lbltrans2 = CriteriaToBinary(POS, NEG, conjunction_judgement_weakly)
subsetter = DefaultSubSetter()
splitter = SubLabelProviderSplitter(subsetter)
min_sample_builder = MinSample.builder(clf_builder_lbl, lbltrans2, splitter, 10)
criteria_wsa_builder = CriteriaWSA.builder(
    clf_builder_lbl_cv,
    full_clf,
    acc_builder,
    clf_builder_lbl,
    clf_builder_lbl_cv,
    subsetter,
    lbltrans2,
    splitter,
    10,
)
skip_wsa_builder = SkipNegWSA.builder(
    clf_builder_lbl_cv,
    full_clf,
    acc_builder,
    clf_builder_lbl,
    clf_builder_lbl_cv,
    subsetter,
    lbltrans2,
    splitter,
    10,
)
random_builder = RandomSampling.builder()
al_builder = TwoPhase.builder(
    llm_prefer_builder,
    skip_wsa_builder,
    SameStateCount.builder(lbltrans2, 25),
)
random_first = TwoPhase.builder(
    random_builder,
    skip_wsa_builder,
    lambda x, y: DocCountStopCritertion(200),
)
orig_autotar_builder = AutoTARReduced.builder(clf_builder, lbltrans, 100, 10)

# %%

# %%
bin_truth = lbltrans2(pred_provider)
len_pos_llm = bin_truth.document_count(POS)
target_path = Path("data/simulation/at/")
target_path.mkdir(parents=True, exist_ok=True)

#%%
for i in range(30):
    plotter_path = target_path / f"run_plotter_{i}.pkl"
    if not plotter_path.exists():
        al = orig_autotar_builder(ds_env.from_environment(ds_env, shared_labels=False), POS, NEG)
        aprioris = {
            f"Perfect{t}": AprioriCriteriaRecallTarget(POS, NEG, lbltrans, target=t / 100)
            for t in range(60, 105, 5)
        }
        stopafter_k = {
            f"Stop_after{k}": SameStateCount.builder(lbltrans2, k)(POS, NEG)
            for k in [50, 100, 150]
        }
        criteria = {
            "Budget": BudgetCriteriaStoppingRule(POS, NEG, lbltrans),
            **aprioris,
            **stopafter_k,
        }
        estimators = dict()
        exp = ExperimentIterator(al, POS, NEG, aprioris, estimators)
        plotter = CriteriaTarPlotter(POS, NEG, lbltrans, dataset_name="FORAS")
        simulator = TarSimulator(exp, plotter, max_it=len_pos_llm)
        simulator.simulate()
        plotter.show()
        plotter_path = target_path / f"run_plotter_{i}.pkl"
        with plotter_path.open("wb") as fh:
            pickle.dump(plotter, fh)

