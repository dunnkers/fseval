from dataclasses import dataclass, field
from typing import Any, Dict

from hydra.core.config_store import ConfigStore
from omegaconf import II, MISSING

from fseval.pipeline.cv import CrossValidatorConfig
from fseval.pipeline.dataset import DatasetConfig
from fseval.pipelines.rank_and_validate import RankAndValidateConfig


@dataclass
class CallbacksConfig:
    stdout: bool = False
    wandb: bool = False


@dataclass
class BaseConfig:
    dataset: DatasetConfig = MISSING
    cv: CrossValidatorConfig = MISSING
    pipeline: Any = MISSING
    callbacks: Dict = field(default_factory=lambda: dict())


cs = ConfigStore.instance()
cs.store(group="pipeline", name="base_rank_and_validate", node=RankAndValidateConfig)
cs.store(name="base_config", node=BaseConfig)
