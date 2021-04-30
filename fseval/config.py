from dataclasses import dataclass, field
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from typing import Dict, Any

@dataclass
class DataSource:
    _target_: str = MISSING
    name: str = MISSING
    identifier: str = MISSING
    task: str = MISSING
    multivariate: bool = False
    misc: Any = None

@dataclass
class CrossValidator:
    _target_: str = MISSING
    n_splits: int = MISSING
    test_size: float = 0.1
    random_state: int = 0
    fold: int = 0

@dataclass
class ExperimentConfig:
    project: str = MISSING
    task: str = MISSING
    datasrc: DataSource = MISSING
    cv: CrossValidator = MISSING # _target_ must be a BaseCrossValidator

cs = ConfigStore.instance()
cs.store(name="config", node=ExperimentConfig)

