from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


class Task(Enum):
    regression = 1
    classification = 2


@dataclass
class DatasetConfig:
    _target_: str = "fseval.datasets.Dataset"
    name: str = MISSING
    task: Task = MISSING
    """
        dataset adapter. must be of fseval.adapters.Adapter type, i.e. must implement a
        get_data() -> (X, y) method. 
    """
    adapter: Any = MISSING
    """ relevant features or instances. """
    relevant_features: Optional[Dict] = None


@dataclass
class CrossValidatorConfig:
    """
    Parameters of both BaseCrossValidator and BaseShuffleSplit.
    """

    _target_: str = "fseval.cv.CrossValidator"
    name: str = MISSING
    """ splitter. must be BaseCrossValidator or BaseShuffleSplit; should at least 
        implement a `split()` and `get_n_splits()` function. """
    splitter: Any = None
    fold: int = 0


@dataclass
class ResampleConfig:
    _target_: str = "fseval.resampling.Resample"
    replace: bool = False
    sample_size: Any = None  # float [0.0 to 1.0] or int [1 to n_samples]
    random_state: Optional[int] = None
    stratify: Optional[List] = None


@dataclass
class RankerConfig:
    _target_: str = "fseval.rankers.Ranker"
    name: str = MISSING
    task: Task = MISSING
    multivariate: bool = False
    """ classifier. must have _target_ of BaseEstimator type with fit() method. """
    classifier: Any = None
    """ regressor. must have _target_ of BaseEstimator type with fit() method. """
    regressor: Any = None


@dataclass
class ValidatorConfig:
    _target_: str = "fseval.validators.Validator"
    name: str = MISSING
    task: Task = MISSING
    multivariate: bool = False
    """ classifier. must have _target_ of BaseEstimator type with fit() method. """
    classifier: Any = None
    """ regressor. must have _target_ of BaseEstimator type with fit() method. """
    regressor: Any = None


@dataclass
class ExperimentConfig:
    _target_: str = "fseval.experiment.Experiment"
    project: str = MISSING
    dataset: DatasetConfig = MISSING
    cv: CrossValidatorConfig = MISSING
    resample: ResampleConfig = MISSING
    ranker: RankerConfig = MISSING
    validator: ValidatorConfig = MISSING


cs = ConfigStore.instance()
cs.store(name="base_config", node=ExperimentConfig)
