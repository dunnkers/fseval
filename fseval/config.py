from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from fseval.pipeline.cv import CrossValidatorConfig
from fseval.pipeline.dataset import DatasetConfig
from fseval.pipeline.estimator import TaskedEstimatorConfig
from fseval.pipeline.resample import ResampleConfig

cs = ConfigStore.instance()


@dataclass
class StorageProviderConfig:
    _target_: str = MISSING
    load_dir: Optional[str] = None
    save_dir: Optional[str] = None


@dataclass
class LocalStorageProviderConfig(StorageProviderConfig):
    ...


@dataclass
class WandbStorageProviderConfig(LocalStorageProviderConfig):
    entity: Optional[str] = None
    project: Optional[str] = None
    run_id: Optional[str] = None


cs.store(
    group="storage_provider",
    name="base_local_storage_provider",
    node=LocalStorageProviderConfig,
)
cs.store(
    group="storage_provider",
    name="base_wandb_storage_provider",
    node=WandbStorageProviderConfig,
)


@dataclass
class BaseConfig:
    pipeline: str = MISSING  # pipeline name
    dataset: DatasetConfig = MISSING
    cv: CrossValidatorConfig = MISSING
    callbacks: Dict[str, Any] = field(default_factory=dict)
    storage_provider: StorageProviderConfig = field(
        default_factory=lambda: StorageProviderConfig(
            _target_="fseval.storage_providers.MockStorageProvider"
        )
    )


cs.store(name="base_config", node=BaseConfig)


@dataclass
class RankAndValidateConfig(BaseConfig):
    _target_: str = "fseval.pipelines.rank_and_validate.BootstrappedRankAndValidate"
    resample: ResampleConfig = MISSING
    ranker: TaskedEstimatorConfig = MISSING
    validator: TaskedEstimatorConfig = MISSING
    n_bootstraps: int = MISSING
    n_jobs: Optional[int] = MISSING
    all_features_to_select: str = MISSING
    upload_ranking_scores: bool = MISSING
    upload_validation_scores: bool = MISSING


cs.store(name="base_rank_and_validate", node=RankAndValidateConfig)
