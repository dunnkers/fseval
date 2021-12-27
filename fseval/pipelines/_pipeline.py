from dataclasses import dataclass

from fseval.pipeline.cv import CrossValidator
from fseval.pipeline.dataset import DatasetLoader
from fseval.pipelines._callback_collection import CallbackCollection
from fseval.types import AbstractStorage
from omegaconf import MISSING


@dataclass
class Pipeline:
    callbacks: CallbackCollection = MISSING
    dataset: DatasetLoader = MISSING
    cv: CrossValidator = MISSING
    storage: AbstractStorage = MISSING
