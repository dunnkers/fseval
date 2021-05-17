#%%
import logging
import os
from abc import ABC, abstractmethod
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
from fseval.pipeline.resample import Resample, ResampleConfig
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import II, MISSING
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.ensemble._base import _BaseHeterogeneousEnsemble
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


class AbstractEstimator(ABC):
    @abstractmethod
    def fit(self, X, y):
        ...

    @abstractmethod
    def transform(self, X, y):
        ...

    @abstractmethod
    def fit_transform(self, X, y):
        ...

    @abstractmethod
    def score(self, X, y) -> pd.DataFrame:
        ...


@dataclass
class AbstractExperiment(AbstractEstimator):
    estimators: List[AbstractEstimator] = field(default_factory=lambda: [])

    def set_estimators(self, estimators: List[AbstractEstimator] = []):
        self.estimators = estimators

    @property
    def _scoring_metadata(self) -> List:
        return []

    def _get_scoring_metadata(self, estimator):
        metadata = {}

        for meta_attribute in self._scoring_metadata:
            metadata[meta_attribute] = getattr(estimator, meta_attribute, None)

        return metadata

    def fit(self, X, y) -> AbstractEstimator:
        for estimator in self.estimators:
            estimator.fit(X, y)

        return self

    def transform(self, X, y):
        ...

    def fit_transform(self, X, y):
        for estimator in self.estimators:
            estimator.fit_transform(X, y)

    def _score_to_dataframe(self, score):
        if isinstance(score, pd.DataFrame):
            return score
        elif isinstance(score, float) or isinstance(score, int):
            return pd.DataFrame([{"score": score}])
        else:
            raise ValueError(f"illegal score type received: {type(score)}")

    def score(self, X, y) -> pd.DataFrame:
        scores = pd.DataFrame()

        for estimator in self.estimators:
            score = estimator.score(X, y)
            score_df = self._score_to_dataframe(score)

            metadata = self._get_scoring_metadata(estimator)
            score_df = score_df.assign(**metadata)

            scores = scores.append(score_df)

        return pd.DataFrame(scores)


# @dataclass
# class RankAndValidateEstimator(Pipeline):
#     callback_list: CallbackList = CallbackList([])
#     validator: Any = MISSING
#     n_features_to_select: int = MISSING

#     def __post_init__(self):
#         self.steps = [
#             ("rank-and-validate", self.ranker),
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
class SubsetValidator(Pipeline):
    callback_list: CallbackList = CallbackList([])
    resample: Resample = MISSING
    ranker: Any = MISSING
    validator: Any = MISSING
    n_features_to_select: int = MISSING

    def _create_selector(self):
        selector = SelectFromModel(
            estimator=self.ranker,
            threshold=-np.inf,
            max_features=self.n_features_to_select,
            prefit=False,
        )
        return selector

    def __post_init__(self):
        self.steps = [
            ("subset_selector", self._create_selector()),
            ("subset_validator", self.validator),
        ]
        self.memory = None
        self.verbose = True

    def fit(self, X, y=None, **fit_params):
        X, y = self.resample.transform(X, y)
        return super(SubsetValidator, self).fit(X, y, **fit_params)


@dataclass
class DatasetValidator(AbstractExperiment):
    callback_list: CallbackList = CallbackList([])
    resample: Resample = MISSING
    ranker: Any = MISSING
    validator: Any = MISSING
    p: int = MISSING

    @property
    def _scoring_metadata(self) -> List:
        return ["n_features_to_select", "resample."]

    def _create_validator(self, n_features_to_select: int):
        validator = SubsetValidator(
            callback_list=self.callback_list,
            resample=self.resample,
            ranker=self.ranker,
            validator=clone(self.validator),
            n_features_to_select=n_features_to_select,
        )
        return validator

    def __post_init__(self):
        n_validations = np.arange(1, min(50, self.p) + 1)
        estimators = [
            self._create_validator(n_features_to_select=i) for i in n_validations
        ]
        self.set_estimators(estimators)


@dataclass
class RankAndValidate(AbstractExperiment, RankAndValidateConfig):
    callback_list: CallbackList = CallbackList([])
    p: int = MISSING

    @property
    def _scoring_metadata(self) -> List:
        return ["p"]

    def _create_validator(self, random_state=None):
        resample = clone(self.resample)
        resample.random_state = random_state

        validator = DatasetValidator(
            callback_list=self.callback_list,
            resample=resample,
            ranker=self.ranker,
            validator=self.validator,
            p=self.p,
        )
        return validator

    def __post_init__(self):
        n_bootstraps = np.arange(1, self.n_bootstraps + 1)
        estimators = [self._create_validator(random_state=i) for i in n_bootstraps]
        self.set_estimators(estimators)

    # def fit(self, X, y):
    #     X, y = self.resample.transform(X, y)
    #     fit_ranker = self.ranker.fit(X, y)
    #     bootstrap_experiment = BootstrapExperiment()
    #     select_subset = SelectFromModel(
    #         estimator=self.ranker,
    #         threshold=-np.inf,
    #         max_features=self.n_features_to_select,
    #         prefit=True,
    #     )
    #     X = select_subset.transform(X)
    #     self.validator = self.validator.fit(X, y)

    # def __post_init__(self):
    #     dataset_validator = lambda random_state: DatasetValidator(
    #         callback_list=self.callback_list,
    #         resample=self._get_resampler(random_state),
    #         ranker=self.ranker,
    #         validator=self.validator,
    #         p=self.p,
    #     )
    #     self.steps = [
    #         (f"bootstrap={random_state}", dataset_validator(random_state))
    #         for random_state in np.arange(1, self.n_bootstraps)
    #     ]
    #     # self.steps = [("dataset-validator", dataset_validator)]
    #     self.memory = None
    #     self.verbose = True


#%%
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


# @dataclass
# class SubsetValidator(BaseEstimator):
#     callback_list: CallbackList = CallbackList([])
#     ranker: Any = MISSING
#     validator: Any = MISSING
#     n_features_to_select: int = MISSING

#     def fit(self, X, y=None):
#         select_subset = SelectFromModel(
#             estimator=self.ranker,
#             threshold=-np.inf,
#             max_features=self.n_features_to_select,
#             prefit=True,
#         )
#         X = select_subset.transform(X)
#         self.validator = self.validator.fit(X, y)
#         return self

#     def score(self, X, y=None):
#         return self.validator.score(X, y)

#     def transform(self, X):
#         return X


# @dataclass
# class DatasetValidator(Pipeline):
#     callback_list: CallbackList = CallbackList([])
#     resample: Resample = MISSING
#     ranker: Any = MISSING
#     validator: Any = MISSING
#     p: int = MISSING

#     def __post_init__(self):
#         # self.ranker = clone(self.ranker)
#         self.validator = clone(self.validator)

#         subset_validator = lambda k: SubsetValidator(
#             callback_list=self.callback_list,
#             # ranker=self.ranker,
#             validator=self.validator,
#             n_features_to_select=k,
#         )
#         self.steps = [
#             (f"subset_validation_k={k}", subset_validator(k))
#             for k in np.arange(1, min(50, self.p) + 1)
#         ]
#         self.memory = None
#         self.verbose = True

#     def _fit(self, X, y=None, **fit_params_steps):
#         X, y = self.resample.transform(X, y)
#         fit_ranker = self.ranker.fit(X, y)

#         for step_name, step_estimator in self.steps:
#             self.set_params(**{f"{step_name}__ranker": fit_ranker})

#         return super(DatasetValidator, self)._fit(X, y, **fit_params_steps)

#     def score(self, X, y=None):
#         X, y = self.resample.transform(X, y)
#         validator_steps = self.steps[1:]

#         total_score = 0
#         for name, validator in validator_steps:
#             score = validator.score(X, y)
#             print(score)
#             total_score += score
#         avg_score = total_score / len(self.steps)
#         print(avg_score)
#         return avg_score


@dataclass
class RankAndValidateOld(Pipeline, RankAndValidateConfig):
    callback_list: CallbackList = CallbackList([])
    p: int = MISSING

    def _get_resampler(self, random_state):
        resampler = clone(self.resample)
        resampler.random_state = random_state
        return resampler

    def __post_init__(self):
        dataset_validator = lambda random_state: DatasetValidator(
            callback_list=self.callback_list,
            resample=self._get_resampler(random_state),
            ranker=self.ranker,
            validator=self.validator,
            p=self.p,
        )
        self.steps = [
            (f"bootstrap={random_state}", dataset_validator(random_state))
            for random_state in np.arange(1, self.n_bootstraps)
        ]
        # self.steps = [("dataset-validator", dataset_validator)]
        self.memory = None
        self.verbose = True

    # def fit(self, X, y=None, **fit_params):
    # for random_bootstrap_state in self.n_bootstraps:
    # self.resample.random_state = random_bootstrap_state
    # X, y = self.resample.transform(X, y)
    # super(RankAndValidate, self).fit(X, y, **fit_params)

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
