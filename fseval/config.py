from dataclasses import dataclass, field
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from typing import Dict, Any, List, Union, Optional


@dataclass
class DatasetConfig:
    _target_: str = MISSING
    name: str = MISSING
    identifier: str = MISSING
    task: str = MISSING
    multivariate: bool = False
    misc: Any = None


@dataclass
class CrossValidatorConfig:
    """
    Parameters of both BaseCrossValidator and BaseShuffleSplit.
    """

    _target_: str = MISSING
    n_splits: int = 5
    shuffle: bool = False
    test_size: Optional[float] = None
    train_size: Optional[float] = None
    random_state: Optional[int] = None
    fold: int = 0


@dataclass
class ResampleConfig:
    replace: bool = True
    n_samples: Union[int, None] = None
    random_state: Union[int, None] = None
    stratify: Union[List, None] = None


@dataclass
class RankerConfig:
    _target_: str = MISSING
    name: str = MISSING
    compatibility: List[str] = field(default_factory=lambda: [])
    n_features_to_select: int = 1


@dataclass
class ExperimentConfig:
    project: str = MISSING
    dataset: DatasetConfig = MISSING
    cv: CrossValidatorConfig = MISSING
    resample: ResampleConfig = MISSING
    ranker: RankerConfig = MISSING
    """ validator must have _target_ of BaseEstimator type. """
    validator: Any = MISSING


cs = ConfigStore.instance()
cs.store(name="config", node=ExperimentConfig)
