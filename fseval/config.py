from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import II, MISSING

from fseval.base import Task
from fseval.cv._cross_validator import CrossValidatorConfig
from fseval.datasets._dataset import DatasetConfig
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
