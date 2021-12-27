from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from fseval.pipeline.cv import CrossValidatorConfig
from fseval.pipeline.dataset import DatasetConfig
from fseval.pipeline.estimator import TaskedEstimatorConfig
from fseval.pipeline.resample import ResampleConfig

cs = ConfigStore.instance()


### Callbacks ###


@dataclass
class StorageConfig:
    _target_: str = MISSING
    load_dir: Optional[str] = None
    save_dir: Optional[str] = None


defaults = [
    "_self_",
    {"dataset": MISSING},
    {"ranker": MISSING},
    {"validator": MISSING},
    {"cv": "kfold"},
    {"storage": "local"},
    {"resample": "shuffle"},
    {"override hydra/job_logging": "colorlog"},
    {"override hydra/hydra_logging": "colorlog"},
]


@dataclass
class BaseConfig:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    _target_: str = "fseval.pipelines.rank_and_validate.BootstrappedRankAndValidate"
    pipeline: str = "rank-and-validate"
    dataset: DatasetConfig = MISSING
    cv: CrossValidatorConfig = MISSING
    callbacks: Dict[str, Any] = field(default_factory=dict)
    storage: StorageConfig = field(
        default_factory=lambda: StorageConfig(_target_="fseval.storage.MockStorage")
    )
    resample: ResampleConfig = MISSING
    ranker: TaskedEstimatorConfig = MISSING
    validator: TaskedEstimatorConfig = MISSING
    n_bootstraps: int = 1
    n_jobs: Optional[int] = 1
    all_features_to_select: str = "range(1, min(50, p) + 1)"
    metrics: Dict[str, Any] = field(default_factory=dict)


cs.store(name="base_config", node=BaseConfig, provider="fseval")
