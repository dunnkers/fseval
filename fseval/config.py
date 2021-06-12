from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from fseval.pipeline.cv import CrossValidatorConfig
from fseval.pipeline.dataset import DatasetConfig
from fseval.pipeline.estimator import TaskedEstimatorConfig
from fseval.pipeline.resample import ResampleConfig


@dataclass
class BaseConfig:
    dataset: DatasetConfig = MISSING
    cv: CrossValidatorConfig = MISSING
    callbacks: Dict[str, Any] = field(default_factory=dict)
    storage_provider: Any = field(
        default_factory=lambda: dict(_target_="fseval.types.AbstractStorageProvider")
    )


cs = ConfigStore.instance()
cs.store(name="base_config", node=BaseConfig)


@dataclass
class RankAndValidateConfig(BaseConfig):
    _target_: str = (
        "fseval.pipelines.rank_and_validate._components.BootstrappedRankAndValidate"
    )
    pipeline: str = MISSING
    resample: ResampleConfig = MISSING
    ranker: TaskedEstimatorConfig = MISSING
    validator: TaskedEstimatorConfig = MISSING
    n_bootstraps: int = MISSING
    n_jobs: Optional[int] = MISSING
    all_features_to_select: str = MISSING


cs.store(name="base_rank_and_validate", node=RankAndValidateConfig)
