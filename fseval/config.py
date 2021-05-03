from dataclasses import dataclass, field
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from typing import Dict, Any, List, Union, Optional
from enum import Enum


class Task(Enum):
    regression = 1
    classification = 2


@dataclass
class DatasetConfig:
    _target_: str = MISSING
    name: str = MISSING
    task: Task = MISSING
    """
        dataset adapter. must be of fseval.adapters.Adapter type, i.e. must implement a
        get_data() -> (X, y) method. 
    """
    adapter: Any = MISSING


@dataclass
class CrossValidatorConfig:
    """
    Parameters of both BaseCrossValidator and BaseShuffleSplit.
    """

    _target_: str = MISSING
    name: str = MISSING
    """ splitter. must be BaseCrossValidator or BaseShuffleSplit; should at least 
        implement a `split()` and `get_n_splits()` function. """
    splitter: Any = None
    fold: int = 0


@dataclass
class ResampleConfig:
    _target_: str = MISSING
    replace: bool = False
    n_samples: Union[int, None] = None
    random_state: Union[int, None] = None
    stratify: Union[List, None] = None


@dataclass
class RankerConfig:
    _target_: str = MISSING
    name: str = MISSING
    task: Task = MISSING
    multivariate: bool = False
    """ classifier. must have _target_ of BaseEstimator type with fit() method. """
    classifier: Any = None
    """ regressor. must have _target_ of BaseEstimator type with fit() method. """
    regressor: Any = None


@dataclass
class ExperimentConfig:
    _target_: str = MISSING
    project: str = MISSING
    dataset: DatasetConfig = MISSING
    cv: CrossValidatorConfig = MISSING
    resample: ResampleConfig = MISSING
    ranker: RankerConfig = MISSING
    """ validator. must have _target_ of BaseEstimator type with fit() method. """
    validator: Any = MISSING


cs = ConfigStore.instance()
cs.store(name="config", node=ExperimentConfig)
