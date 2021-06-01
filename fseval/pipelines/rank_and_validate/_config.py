import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
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

from fseval.pipeline.cv import CrossValidator
from fseval.pipeline.dataset import Dataset, DatasetConfig
from fseval.pipeline.estimator import Estimator, TaskedEstimatorConfig
from fseval.pipeline.resample import Resample, ResampleConfig
from fseval.types import AbstractEstimator, Callback, Task

from .._experiment import Experiment
from .._pipeline import Pipeline

logger = logging.getLogger(__name__)


@dataclass
class RankAndValidateConfig:
    _target_: str = (
        "fseval.pipelines.rank_and_validate._components.BootstrappedRankAndValidate"
    )
    name: str = "rank-and-validate"
    resample: ResampleConfig = MISSING
    ranker: TaskedEstimatorConfig = MISSING
    validator: TaskedEstimatorConfig = MISSING
    n_bootstraps: int = MISSING


@dataclass
class RankAndValidatePipeline(Pipeline):
    """Instantiated version of `RankAndValidateConfig`: the actual pipeline
    implementation."""

    name: str = MISSING
    resample: Resample = MISSING
    ranker: Estimator = MISSING
    validator: Estimator = MISSING
    n_bootstraps: int = MISSING

    def _get_config(self):
        return {
            key: getattr(self, key)
            for key in RankAndValidatePipeline.__dataclass_fields__.keys()
        }
