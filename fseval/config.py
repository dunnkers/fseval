from dataclasses import dataclass, field
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from typing import Dict, Any, List, Union

@dataclass
class DataSource:
    _target_: str = MISSING
    name: str = MISSING
    identifier: str = MISSING
    task: str = MISSING
    multivariate: bool = False
    misc: Any = None

# @dataclass
# class CrossValidator:
#     _target_: str = MISSING
#     n_splits: int = MISSING
#     test_size: float = 0.1
#     random_state: int = 0
#     fold: int = 0

@dataclass
class Bootstrap:
    replace: bool = True
    n_samples: Union[int, None] = None
    random_state: Union[int, None] = None
    stratify: Union[List, None] = None

@dataclass
class ExperimentConfig:
    project: str = MISSING
    datasrc: DataSource = MISSING
    cv: Any = MISSING # _target_ must be a BaseCrossValidator
    cv_fold: int = 0
    task: str = MISSING
    bootstrap: Bootstrap = MISSING

cs = ConfigStore.instance()
cs.store(name="config", node=ExperimentConfig)

