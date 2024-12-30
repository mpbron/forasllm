import random
from typing import Any

import instancelib as il
import numpy as np
import pandas as pd
from allib.environment.base import AbstractEnvironment
from allib.machinelearning.taroptimized import ALSklearn
from allib.typehints.typevars import DT, IT, KT, LT, RT, VT
from cleanlab.classification import CleanLearning
from sklearn.preprocessing import LabelEncoder


def noisy_label_correction(
    env: AbstractEnvironment[IT, KT, DT, VT, RT, LT],
    noisy_labels: il.LabelProvider[KT, LT],
    pos_label: LT,
    neg_label: LT,
    cl: CleanLearning,
    insclf: ALSklearn[IT, KT, DT, VT, LT, Any],
    scale_factor: float = 2,
) -> il.InstanceProvider[IT, KT, DT, VT, RT]:
    # Model specification
    # type: ignore

    # Get prediction data
    poss = env.get_subset_by_labels(env.dataset, pos_label, labelprovider=noisy_labels)
    negs = env.get_subset_by_labels(env.dataset, neg_label, labelprovider=noisy_labels)
    negs_sampled = env.create_bucket(
        random.sample(negs.key_list, round(len(poss) * scale_factor))
    )
    sample = env.combine(poss, negs_sampled)
    sample_instances = list(sample.get_all())
    sample_keys = [ins.identifier for ins in sample_instances]
    llm_labels = list(env.to_pandas(sample, noisy_labels)["label"].values)
    llm_mat = insclf.vectorizer.transform(sample_instances)
    encoder = LabelEncoder()
    encoder.fit(llm_labels)
    elabels = encoder.transform(llm_labels)

    label_issues = cl.find_label_issues(X=llm_mat, labels=elabels)
    lowest_quality_labels = label_issues["label_quality"].argsort().to_numpy()

    def combine_data(index):
        return pd.DataFrame(
            {
                "key": sample_keys,
                "given_label": llm_labels,
                "quality": label_issues["label_quality"],
                "predicted_label": encoder.inverse_transform(
                    label_issues["predicted_label"]
                ),
            },
        ).iloc[index]

    lowqual = combine_data(lowest_quality_labels)
    highqual = combine_data(lowest_quality_labels[::-1])

    possible_missing = lowqual[
        np.logical_and(
            lowqual["given_label"] == neg_label, lowqual["predicted_label"] == pos_label
        )
    ]
    probable_relevant = highqual[
        np.logical_and(
            highqual["given_label"] == pos_label,
            highqual["predicted_label"] == pos_label,
        )
    ]
    probable_irrelevant = lowqual[
        np.logical_and(
            lowqual["given_label"] == pos_label,
            lowqual["predicted_label"] == neg_label,
        )
    ]
    skip_prob_irrelevant = env.create_bucket(
        frozenset(poss).difference(
            random.sample(
                list(probable_irrelevant.key), len(probable_irrelevant.key) // 2
            )
        )
    )
    reviewed = env.create_bucket(
        np.concatenate((probable_relevant.key, possible_missing.key))
    )
    unreviewed = env.create_bucket(frozenset(env.dataset).difference(reviewed))

    insclf.fit_provider(sample, noisy_labels)
    corrected_labels = il.MemoryLabelProvider.from_tuples(insclf.predict(unreviewed))
    review_candidates = env.combine(
        reviewed,
        env.get_subset_by_labels(unreviewed, pos_label, labelprovider=corrected_labels),
    )
    if len(review_candidates) < len(poss):
        return review_candidates
    return skip_prob_irrelevant


def find_likely_relevant(
    env: AbstractEnvironment[IT, KT, DT, VT, RT, LT],
    noisy_labels: il.LabelProvider[KT, LT],
    pos_label: LT,
    neg_label: LT,
    cl: CleanLearning,
    insclf: ALSklearn,
    scale_factor: float = 2,
) -> il.InstanceProvider[IT, KT, DT, VT, RT]:
    # Model specification
    # type: ignore

    # Get prediction data
    poss = env.get_subset_by_labels(env.dataset, pos_label, labelprovider=noisy_labels)
    negs = env.get_subset_by_labels(env.dataset, neg_label, labelprovider=noisy_labels)
    negs_sampled = env.create_bucket(
        random.sample(negs.key_list, round(len(poss) * scale_factor))
    )
    sample = env.combine(poss, negs_sampled)
    sample_instances = list(sample.get_all())
    sample_keys = [ins.identifier for ins in sample_instances]
    llm_labels = list(env.to_pandas(sample, noisy_labels)["label"].values)
    llm_mat = insclf.vectorizer.transform(sample_instances)
    encoder = LabelEncoder()
    encoder.fit(llm_labels)
    elabels = encoder.transform(llm_labels)

    label_issues = cl.find_label_issues(X=llm_mat, labels=elabels)
    lowest_quality_labels = label_issues["label_quality"].argsort().to_numpy()

    def combine_data(index):
        return pd.DataFrame(
            {
                "key": sample_keys,
                "given_label": llm_labels,
                "quality": label_issues["label_quality"],
                "predicted_label": encoder.inverse_transform(
                    label_issues["predicted_label"]
                ),
            },
        ).iloc[index]

    lowqual = combine_data(lowest_quality_labels)
    highqual = combine_data(lowest_quality_labels[::-1])

    probable_relevant = highqual[
        np.logical_and(
            highqual["given_label"] == pos_label,
            highqual["predicted_label"] == pos_label,
        )
    ]

    likely_relevant = env.create_bucket(probable_relevant.key)
    return likely_relevant
