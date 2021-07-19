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
class StorageConfig:
    _target_: str = MISSING
    load_dir: Optional[str] = None
    save_dir: Optional[str] = None


@dataclass
class LocalStorageConfig(StorageConfig):
    ...


@dataclass
class WandbStorageConfig(LocalStorageConfig):
    entity: Optional[str] = None
    project: Optional[str] = None
    run_id: Optional[str] = None


cs.store(
    group="storage",
    name="base_local_storage",
    node=LocalStorageConfig,
)
cs.store(
    group="storage",
    name="base_wandb_storage",
    node=WandbStorageConfig,
)


@dataclass
class BaseConfig:
    pipeline: str = MISSING  # pipeline name
    dataset: DatasetConfig = MISSING
    cv: CrossValidatorConfig = MISSING
    callbacks: Dict[str, Any] = field(default_factory=dict)
    storage: StorageConfig = field(
        default_factory=lambda: StorageConfig(_target_="fseval.storage.MockStorage")
    )


cs.store(name="base_config", node=BaseConfig)


# @dataclass
# class MetricsConfig:
#     """Configuration concerning the pipeline metrics. Parameters:

#     - `bootstrap`: Dict[str, Any] - Aggregated metrics for the
#         bootstrapped pipeline.
#     - `pipeline`: Dict[str, Any] - Aggregated metrics for the pipeline.
#     - `ranking`: Dict[str, Any] - Metrics for validating a feature ranking, e.g. using a
#         ground-truth.
#     - `support`: Dict[str, Any] - Metrics for validating a feature support vector. e.g.,
#         this is an array indicating yes/no which features to include in a feature
#         subset. The array is validated by running the validation estimator on this
#         feature subset.
#     - `dataset`: Dict[str, Any] - Aggregated metrics for all feature subsets. e.g. 50
#         feature subsets for p >= 50.
#     - `subset`: Dict[str, Any] - Metrics for validation estimator. Validates 1 feature subset.
#     """

#     bootstrap: Dict[str, Any] = field(default_factory=dict)
#     pipeline: Dict[str, Any] = field(default_factory=dict)
#     ranking: Dict[str, Any] = field(default_factory=dict)
#     support: Dict[str, Any] = field(default_factory=dict)
#     dataset: Dict[str, Any] = field(default_factory=dict)
#     subset: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RankAndValidateConfig(BaseConfig):
    _target_: str = "fseval.pipelines.rank_and_validate.BootstrappedRankAndValidate"
    resample: ResampleConfig = MISSING
    ranker: TaskedEstimatorConfig = MISSING
    validator: TaskedEstimatorConfig = MISSING
    n_bootstraps: int = MISSING
    n_jobs: Optional[int] = MISSING
    all_features_to_select: str = MISSING
    metrics: Dict[str, Any] = field(default_factory=dict)
    upload_ranking_scores: bool = MISSING
    upload_validation_scores: bool = MISSING


cs.store(name="base_rank_and_validate", node=RankAndValidateConfig)
