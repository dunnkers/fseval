from dataclasses import dataclass, field
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from typing import Dict, Any, List, Union


@dataclass
class DatasetConfig:
    _target_: str = MISSING
    name: str = MISSING
    identifier: str = MISSING
    task: str = MISSING
    multivariate: bool = False
    misc: Any = None


@dataclass
class BootstrapConfig:
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
    cv: Any = MISSING  # _target_ must be a BaseCrossValidator
    cv_fold: int = 0
    bootstrap: BootstrapConfig = MISSING
    ranker: RankerConfig = MISSING
    validator: Any = MISSING # _target_ must be a BaseEstimator


cs = ConfigStore.instance()
cs.store(name="config", node=ExperimentConfig)
