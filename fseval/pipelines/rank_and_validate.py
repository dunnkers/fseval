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
from fseval.base import AbstractEstimator, Task
from fseval.callbacks._callback import CallbackList
from fseval.cv._cross_validator import CrossValidatorConfig
from fseval.datasets._dataset import Dataset, DatasetConfig
from fseval.pipeline.estimator import Estimator, TaskedEstimatorConfig
from fseval.pipeline.experiment import AbstractExperiment
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
class RankerConfig(TaskedEstimatorConfig):
    _target_class_: str = "fseval.pipelines.rank_and_validate.Ranker"
    dataset: Dict = II("dataset")


@dataclass
class Ranker(Estimator, TaskedEstimatorConfig):
    dataset: Dataset = MISSING

    def score(self, X, y):
        # FIXME don't load dataset twice
        self.dataset.load(ensure_positive_X=True)
        X_importances = self.dataset.get_feature_importances()
        if X_importances is not None:
            assert np.ndim(X_importances) == 1, "instance-based not supported yet."
            ranking = self.feature_importances_

            # mean absolute error
            y_true = X_importances
            y_pred = ranking
            mae = mean_absolute_error(y_true, y_pred)

            # log loss
            y_true = X_importances > 0
            y_pred = ranking
            log_loss_score = log_loss(y_true, y_pred, labels=[0, 1])

            return pd.DataFrame(
                [{"mean_absolute_error": mae, "log_loss": log_loss_score}]
            )
        else:
            return pd.DataFrame()


@dataclass
class RankAndValidateConfig:
    _target_: str = "fseval.pipelines.rank_and_validate.RankAndValidate"
    name: str = "rank-and-validate"
    resample: ResampleConfig = MISSING
    n_bootstraps: int = MISSING

    ranker: RankerConfig = MISSING  # CLI: estimator@pipeline.ranker=chi2
    validator: TaskedEstimatorConfig = MISSING  # CLI: estimator@pipeline.ranker=chi2


@dataclass
class SubsetValidator(Pipeline):  # TODO convert to AbstractExperiment
    callback_list: CallbackList = CallbackList([])
    resample: Resample = MISSING
    ranker: Estimator = MISSING
    validator: Estimator = MISSING
    # overrides
    n_features_to_select: int = MISSING
    bootstrap_state: int = MISSING
    # TODO create "overrides" dict: store exactly these things you want to override in
    # there.

    def __post_init__(self):
        self.selector = SelectFromModel(
            estimator=self.ranker,
            threshold=-np.inf,
            max_features=self.n_features_to_select,
            prefit=True,
        )

        self.steps = [
            ("subset_validator", self.validator),
        ]
        self.memory = None
        self.verbose = False

    def fit(self, X, y=None, **fit_params):
        self.resample.random_state = self.bootstrap_state
        X, y = self.resample.transform(X, y)
        X = self.selector.transform(X)
        return super(SubsetValidator, self).fit(X, y, **fit_params)

    def score(self, X, y=None, **fit_params):
        self.resample.random_state = self.bootstrap_state
        X, y = self.resample.transform(X, y)
        X = self.selector.transform(X)
        return super(SubsetValidator, self).score(X, y, **fit_params)


@dataclass
class DatasetValidator(AbstractExperiment):
    callback_list: CallbackList = CallbackList([])
    resample: Resample = MISSING
    ranker: Estimator = MISSING
    validator: Estimator = MISSING
    # overrides
    p: int = MISSING
    bootstrap_state: int = MISSING

    @property
    def _scoring_metadata(self) -> List:
        return ["n_features_to_select", "bootstrap_state", "fit_time"]

    def _create_validator(self, n_features_to_select: int):
        validator = SubsetValidator(
            callback_list=self.callback_list,
            resample=self.resample,
            ranker=self.ranker,
            validator=clone(self.validator),
            n_features_to_select=n_features_to_select,
            bootstrap_state=self.bootstrap_state,
        )
        return validator

    def __post_init__(self):
        n_validations = np.arange(1, min(50, self.p) + 1)
        estimators = [
            self._create_validator(n_features_to_select=i) for i in n_validations
        ]
        self.set_estimators(estimators)

    def score(self, X, y):
        scores = super(DatasetValidator, self).score(X, y)

        self.callback_list.on_metrics(scores)

        return scores


@dataclass
class RankingValidator(AbstractExperiment):
    callback_list: CallbackList = CallbackList([])
    resample: Resample = MISSING
    ranker: Estimator = MISSING
    validator: Estimator = MISSING
    # overrides
    p: int = MISSING
    bootstrap_state: int = MISSING

    def __post_init__(self):
        validator = DatasetValidator(
            callback_list=self.callback_list,
            resample=self.resample,
            ranker=self.ranker,
            validator=self.validator,
            p=self.p,
            bootstrap_state=self.bootstrap_state,
        )
        self.estimators = [
            self.ranker,
            validator,
        ]


@dataclass
class RankAndValidate(AbstractExperiment, RankAndValidateConfig):
    callback_list: CallbackList = CallbackList([])
    # overrides
    p: int = MISSING

    def _create_validator(self, bootstrap_state=None):
        validator = RankingValidator(
            callback_list=self.callback_list,
            resample=self.resample,
            ranker=self.ranker,
            validator=self.validator,
            p=self.p,
            bootstrap_state=bootstrap_state,
        )
        return validator

    def __post_init__(self):
        n_bootstraps = np.arange(1, self.n_bootstraps + 1)
        estimators = [self._create_validator(bootstrap_state=i) for i in n_bootstraps]
        self.set_estimators(estimators)

    def score(self, X, y):
        scores = super(RankAndValidate, self).score(X, y)
        all_bootstraps = scores.groupby("n_features_to_select")
        result = pd.DataFrame(
            {
                "score_mean": all_bootstraps["score"].mean(),
                "score_std": all_bootstraps["score"].std(),
                "fit_time_mean": all_bootstraps["fit_time"].mean(),
                "fit_time_std": all_bootstraps["fit_time"].std(),
            }
        )

        return result
