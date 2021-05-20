import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from fseval.pipeline.cv import CrossValidator
from fseval.pipeline.dataset import Dataset, DatasetConfig
from fseval.pipeline.estimator import Estimator, TaskedEstimatorConfig
from fseval.pipeline.resample import Resample, ResampleConfig
from fseval.types import AbstractEstimator, Callback, Task
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import II, MISSING
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.ensemble._base import _BaseHeterogeneousEnsemble
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.metrics import log_loss, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import _print_elapsed_time
from sklearn.utils.metaestimators import _BaseComposition
from tqdm import tqdm

from .._experiment import Experiment
from .._pipeline import Pipeline
from ._ranker import Ranker, RankerConfig


@dataclass
class RankAndValidatePipeline(Pipeline):
    name: str = MISSING
    resample: Resample = MISSING
    ranker: Ranker = MISSING
    validator: Estimator = MISSING
    n_bootstraps: int = MISSING

    def _get_config(self):
        return {
            key: getattr(self, key)
            for key in RankAndValidatePipeline.__dataclass_fields__.keys()
        }


@dataclass
class SubsetValidator(Experiment, RankAndValidatePipeline):
    n_features_to_select: int = MISSING
    bootstrap_state: int = MISSING

    def _prepare_data(self, X, y):
        # resample
        self.resample.random_state = self.bootstrap_state
        X, y = self.resample.transform(X, y)

        # select n features
        selector = SelectFromModel(
            estimator=self.ranker,
            threshold=-np.inf,
            max_features=self.n_features_to_select,
            prefit=True,
        )
        X = selector.transform(X)

        return X, y

    def _get_estimator(self):
        yield self.validator

    def score(self, X, y):
        score = super(SubsetValidator, self).score(X, y)
        score["n_features_to_select"] = self.n_features_to_select
        score["fit_time"] = self._fit_time_elapsed
        return score


@dataclass
class DatasetValidator(Experiment, RankAndValidatePipeline):
    bootstrap_state: int = MISSING

    @property
    def _scoring_metadata(self) -> List:
        return ["n_features_to_select", "bootstrap_state", "fit_time"]

    def _get_estimator(self):
        for n_features_to_select in np.arange(1, min(50, self.dataset.p) + 1):
            config = self._get_config()
            validator = config.pop("validator")

            yield SubsetValidator(
                **config,
                validator=clone(validator),
                n_features_to_select=n_features_to_select,
                bootstrap_state=self.bootstrap_state,
            )

    def score(self, X, y):
        scores = super(DatasetValidator, self).score(X, y)

        self.callbacks.on_metrics(scores)

        return scores


@dataclass
class RankingValidator(Experiment, RankAndValidatePipeline):
    bootstrap_state: int = MISSING

    def _get_estimator(self):
        # give ranker access to dataset
        self.ranker.dataset = self.dataset

        # first fit ranker, then run all validations
        return [
            self.ranker,
            DatasetValidator(
                **self._get_config(), bootstrap_state=self.bootstrap_state
            ),
        ]

    def score(self, X, y):
        scores = super(RankingValidator, self).score(X, y)
        scores["bootstrap_state"] = self.bootstrap_state
        return scores


@dataclass
class RankAndValidate(Experiment, RankAndValidatePipeline):
    def _get_estimator(self):
        for bootstrap_state in np.arange(1, self.n_bootstraps + 1):
            yield RankingValidator(
                **self._get_config(),
                bootstrap_state=bootstrap_state,
            )

    def score(self, X, y):
        scores = super(RankAndValidate, self).score(X, y)
        result = pd.DataFrame()

        ranking_result = pd.DataFrame(
            [{
                "r2_score_mean": scores["r2_score"].mean(),
                "r2_score_std": scores["r2_score"].std(),
                "log_loss_mean": scores["log_loss"].mean(),
                "log_loss_std": scores["log_loss"].std(),
            }]
        )
        result = result.append(ranking_result)

        by_dimension = scores.groupby("n_features_to_select")
        validation_result = pd.DataFrame(
            {
                "score_mean": by_dimension["score"].mean(),
                "score_std": by_dimension["score"].std(),
                "fit_time_mean": by_dimension["fit_time"].mean(),
                "fit_time_std": by_dimension["fit_time"].std(),
            }
        )
        result = result.append(validation_result)

        return result
