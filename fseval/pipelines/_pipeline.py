from dataclasses import dataclass, field
from typing import Dict

from omegaconf import MISSING

from fseval.pipeline.cv import CrossValidator
from fseval.pipeline.dataset import DatasetLoader
from fseval.types import AbstractStorageProvider, Callback


@dataclass
class Pipeline:
    callbacks: Dict[str, Callback] = field(default_factory=dict)
    dataset: DatasetLoader = MISSING
    cv: CrossValidator = MISSING
    storage_provider: AbstractStorageProvider = MISSING
