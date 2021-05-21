import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fseval.pipeline.cv import CrossValidator
from fseval.pipeline.dataset import Dataset, DatasetConfig
from fseval.pipeline.estimator import Estimator, TaskedEstimatorConfig
from fseval.pipeline.resample import Resample, ResampleConfig
from fseval.types import AbstractEstimator, Task
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import II, MISSING
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.ensemble._base import _BaseHeterogeneousEnsemble
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.metrics import log_loss, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import _print_elapsed_time
from sklearn.utils.metaestimators import _BaseComposition
from tqdm import tqdm

from .._callback_list import CallbackList
from .._experiment import Experiment


@dataclass
class RankerConfig(TaskedEstimatorConfig):
    _target_class_: str = "fseval.pipelines.rank_and_validate._ranker.Ranker"


@dataclass
class Ranker(Estimator):
    dataset: Optional[Dataset] = None

    def score(self, X, y):
        assert self.dataset is not None, "must set dataset before scoring ranker"

        X_importances = self.dataset.feature_importances
        if X_importances is not None:
            assert np.ndim(X_importances) == 1, "instance-based not supported yet."
            ranking = self.feature_importances_

            # mean absolute error
            y_true = X_importances
            y_pred = ranking
            r2 = r2_score(y_true, y_pred)

            # log loss
            y_true = X_importances > 0
            y_pred = ranking
            log_loss_score = log_loss(y_true, y_pred, labels=[0, 1])

            return pd.DataFrame(
                [
                    {
                        "r2_score": r2,
                        "log_loss": log_loss_score,
                        "fit_time": self._fit_time_elapsed,
                    }
                ]
            )
        else:
            return pd.DataFrame()

    @property
    def feature_importances_(self):
        summation = self.estimator.feature_importances_
        return np.asarray(self.estimator.feature_importances_) / summation
