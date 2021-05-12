import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from fseval.base import Configurable
from fseval.cv import CrossValidator
from fseval.datasets import Dataset
from fseval.resampling import Resample

from ._callbacks import CallbackList

logger = logging.getLogger(__name__)


class PipelineComponent(ABC):
    @abstractmethod
    def run(self, input: Any, callback_list: CallbackList) -> Any:
        ...


@dataclass
class Pipeline(PipelineComponent, Configurable):
    name: str
    dataset: Dataset
    cv: CrossValidator
    resample: Resample

    def run_pipeline(self, callbacks: List = []):
        callback_list = CallbackList(callbacks)

        callback_list.on_pipeline_begin()
        self.run(None, callback_list)
        callback_list.on_pipeline_end()

    def run(self, input: Any, callback_list: CallbackList) -> Any:
        ...
