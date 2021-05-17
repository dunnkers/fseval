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
from fseval.datasets._dataset import DatasetConfig
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
class RankAndValidateConfig:
    _target_: str = "fseval.pipelines.rank_and_validate.RankAndValidate"
    name: str = "rank-and-validate"
    resample: ResampleConfig = MISSING
    n_bootstraps: int = MISSING

    ranker: TaskedEstimatorConfig = MISSING  # CLI: estimator@pipeline.ranker=chi2
    validator: TaskedEstimatorConfig = MISSING  # CLI: estimator@pipeline.ranker=chi2


@dataclass
class SubsetValidator(Pipeline):
    callback_list: CallbackList = CallbackList([])
    resample: Resample = MISSING
    ranker: Estimator = MISSING
    validator: Estimator = MISSING
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
    ranker: Estimator = MISSING
    validator: Estimator = MISSING
    p: int = MISSING

    @property
    def _scoring_metadata(self) -> List:
        return ["n_features_to_select", "resample__random_state"]

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
