from dataclasses import dataclass

from fseval.pipeline.cv import CrossValidator
from fseval.pipeline.dataset import Dataset
from fseval.types import AbstractStorageProvider, Callback


@dataclass
class Pipeline:
    callbacks: Callback
    dataset: Dataset
    cv: CrossValidator
    storage_provider: AbstractStorageProvider
