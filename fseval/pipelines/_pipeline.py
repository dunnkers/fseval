from dataclasses import dataclass

from omegaconf import MISSING

from fseval.pipeline.cv import CrossValidator
from fseval.pipeline.dataset import DatasetLoader
from fseval.pipelines._callback_collection import CallbackCollection
from fseval.types import AbstractStorage


@dataclass
class Pipeline:
    callbacks: CallbackCollection = MISSING
    dataset: DatasetLoader = MISSING
    cv: CrossValidator = MISSING
    storage: AbstractStorage = MISSING
