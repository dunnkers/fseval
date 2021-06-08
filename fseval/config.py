from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from fseval.pipeline.cv import CrossValidatorConfig
from fseval.pipeline.dataset import DatasetConfig
from fseval.pipelines.rank_and_validate import RankAndValidateConfig


@dataclass
class BaseConfig:
    dataset: DatasetConfig = MISSING
    cv: CrossValidatorConfig = MISSING
    pipeline: Any = MISSING
    callbacks: Dict = field(default_factory=lambda: dict())
    storage_provider: Any = field(
        default_factory=lambda: dict(
            _target_="fseval.storage_providers.mock.MockStorageProvider"
        )
    )


cs = ConfigStore.instance()
cs.store(group="pipeline", name="base_rank_and_validate", node=RankAndValidateConfig)
cs.store(name="base_config", node=BaseConfig)
