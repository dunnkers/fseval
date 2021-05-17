import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd
from fseval.base import Task
from fseval.callbacks._callback import CallbackList
from fseval.cv._cross_validator import CrossValidatorConfig
from fseval.datasets._dataset import DatasetConfig
from fseval.pipeline.estimator import TaskedEstimatorConfig
from fseval.pipeline.resample import ResampleConfig
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import II, MISSING
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.metrics import log_loss, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import _print_elapsed_time
from sklearn.utils.metaestimators import _BaseComposition
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class RankAndValidateConfig:
    _target_: str = "fseval.pipelines.rank_and_validate.RankAndValidate"
    name: str = "rank-and-validate"
    resample: ResampleConfig = MISSING
    n_bootstraps: int = MISSING

    ranker: TaskedEstimatorConfig = MISSING  # CLI: estimator@pipeline.ranker=chi2
    validator: TaskedEstimatorConfig = MISSING  # CLI: estimator@pipeline.ranker=chi2


# @dataclass
# class SubsetValidator(Pipeline):
#     callback_list: CallbackList = CallbackList([])
#     validator: Any = MISSING
#     n_features_to_select: int = MISSING

#     def __post_init__(self):
#         self.steps = [
#             ("validator", self.validator),
#         ]
#         self.memory = None
#         self.verbose = False

#     def fit(self, X, y=None, validator__ranker=None):
#         select_subset = SelectFromModel(
#             estimator=ranker,
#             threshold=-np.inf,
#             max_features=self.n_features_to_select,
#             prefit=True,
#         )
#         X = select_subset.transform(X)
#         return super(SubsetValidator, self).fit(X, y)


@dataclass
class SubsetValidator:
    callback_list: CallbackList = CallbackList([])
    ranker: Any = MISSING
    validator: Any = MISSING
    n_features_to_select: int = MISSING

    def fit(self, X, y=None):
        select_subset = SelectFromModel(
            estimator=self.ranker,
            threshold=-np.inf,
            max_features=self.n_features_to_select,
            prefit=True,
        )
        X = select_subset.transform(X)
        self.validator = self.validator.fit(X, y)
        return self

    def score(self, X, y=None):
        return self.validator.score(X, y)

    def transform(self, X):
        return X


@dataclass
class DatasetValidator(Pipeline):
    callback_list: CallbackList = CallbackList([])
    ranker: Any = MISSING
    validator: Any = MISSING
    p: int = MISSING

    def __post_init__(self):
        self.validator = clone(self.validator)

        subset_validator = lambda k: SubsetValidator(
            callback_list=self.callback_list,
            ranker=self.ranker,
            validator=self.validator,
            n_features_to_select=k,
        )
        self.steps = [
            (f"validate-subset (k={k})", subset_validator(k))
            for k in np.arange(1, min(50, self.p) + 1)
        ]
        self.memory = None
        self.verbose = True

    def fit(self, X, y=None):
        ranker = self.ranker.fit(X, y)
        return super(DatasetValidator, self).fit(X, y)

        # self.steps = list(self.steps)

        # ranking_step = self.steps[0]
        # validator_steps = self.steps[1:]

        # with _print_elapsed_time("DatasetValidator", "ranker fit"):
        #     name, ranker = ranking_step
        #     fit_ranker = ranker.fit(X, y)
        #     self.steps[0] = (name, fit_ranker)

        # for step_idx, (name, validator) in enumerate(validator_steps, start=1):
        #     msg = self._log_message(step_idx)
        #     with _print_elapsed_time("DatasetValidator", msg):

        #         validator = clone(validator)
        #         fit_validator = validator.fit(X, y)
        #         self.steps[step_idx] = (name, fit_validator)

        # return self

    def score(self, X, y=None):
        validator_steps = self.steps[1:]

        total_score = 0
        for name, validator in validator_steps:
            score = validator.score(X, y)
            total_score += score
        avg_score = total_score / len(self.steps)
        print(avg_score)
        return avg_score


@dataclass
class RankAndValidate(Pipeline, RankAndValidateConfig):
    callback_list: CallbackList = CallbackList([])

    def __post_init__(self):
        dataset_validator = DatasetValidator(
            callback_list=self.callback_list,
            ranker=self.ranker,
            validator=self.validator,
            p=4,
        )
        self.steps = [("dataset-validator", dataset_validator)]
        self.memory = None
        self.verbose = True

    def fit(self, X, y=None, **fit_params):
        # for random_bootstrap_state in self.n_bootstraps:
        # self.resample.random_state = random_bootstrap_state
        # X, y = self.resample.transform(X, y)
        super(RankAndValidate, self).fit(X, y, **fit_params)

    # resample: Resample
    # ranker: Any = None
    # validator: Any = None
    # n_bootstraps: int = 1

    # def run(self, callback_list: CallbackList) -> Any:
    #     ...
    # # load dataset
    # self.dataset.load()

    # # send runtime properties to callbacks
    # callback_list.on_pipeline_config_update(
    #     {
    #         "dataset": {
    #             "n": self.dataset.n,
    #             "p": self.dataset.p,
    #             "multioutput": self.dataset.multioutput,
    #         }
    #     }
    # )

    # # cross-validation split
    # train_index, test_index = self.cv.get_split(self.dataset.X)

    # # cross-validation subsets
    # X_train, X_test, y_train, y_test = self.dataset.get_subsets(
    #     train_index, test_index
    # )

    # # define pipeline components
    # subset_loader = SubsetLoaderPipe(self.dataset, self.cv)
    # feature_ranker = FeatureRankingPipe(self.ranker)
    # run_estimator = RunEstimatorPipe(self.estimator)
    # ranking_validator = RankingValidator(run_estimator)

    # # load dataset
    # X_train, X_test, y_train, y_test = subset_loader.run()

    # check whether ranker / estimator support this dataset
    # if self.dataset.multioutput:
    #     msg = f"does not support the multioutput datasets ({self.dataset.name})."

    #     assert self.ranker._get_tags().get("multioutput", False), f"ranker {msg}"
    #     assert "multioutput" in self.estimator._get_tags().get(
    #         "multioutput", False
    #     ), f"estimator {msg}"

    # all_scores = []
    # bootstraps = list(range(self.n_bootstraps))
    # p = cast(int, self.dataset.p)
    # for i in bootstraps:
    #     callback_list.on_log(f"running bootstrap #{i}: resample.random_state={i}")

    #     # resample
    #     self.resample.random_state = i
    #     X_train, y_train = self.resample.transform(X_train, y_train)

    #     # feature ranking
    #     data = (X_train, X_test, y_train, y_test)
    #     ranking, fit_time = feature_ranker.run(data, callback_list)
    #     ranker_log = {"ranker_fit_time": fit_time, "resample.random_state": i}

    #     callback_list.on_log(f"ranking fit time: {fit_time:2f}")
    #     callback_list.on_metrics(ranker_log)

    #     # validation
    #     scores = ranking_validator.run((ranking, data), callback_list)
    #     scores["bootstrap"] = i
    #     best_score_index = scores["estimator_score"].argmax()
    #     best_k = scores.iloc[best_score_index]
    #     callback_list.on_log(
    #         f"bootstrap {i} best score: {best_k['estimator_score']} (k={best_k['k']})"
    #     )
    #     all_scores.append(scores)

    # # average score
    # avg_estimator_scores = np.mean(all_scores, axis=0)
    # avg_estimator_scores_dict = [{} for avg_estimator_score in avg_estimator_scores]
    # callback_list.on_metrics(
    #     {"avg_estimator_scores": avg_estimator_scores, "bootstrap": bootstraps}
    # )

    # # summary
    # best_k_index = np.argmax(avg_estimator_scores)
    # best_k = int(k_best[best_k_index])
    # best_k_score = float(avg_estimator_scores[best_k_index])
    # callback_list.on_summary({"best_k": best_k, "best_k_score": best_k_score})
