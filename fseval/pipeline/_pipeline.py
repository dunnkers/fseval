import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, List

from fseval.base import ConfigurableEstimator
from fseval.config import PipelineConfig
from fseval.cv import CrossValidator
from fseval.datasets import Dataset
from fseval.resampling import Resample

from ._callbacks import Callback

logger = logging.getLogger(__name__)


class Component(ABC):
    @abstractmethod
    def run(self, input) -> Any:
        ...


@dataclass
class Pipeline(Component):
    dataset: Dataset
    cv: CrossValidator
    resample: Resample
    estimator: ConfigurableEstimator
    pipeline: PipelineConfig

    # pipeline component working directory. save files to this dir; wandb will sync them.
    # run_dir: str
    # executes these components in order
    # components: List[Component] = field(default_factory=lambda: [])
    # callbacks
    # callbacks: List[Callback] = field(default_factory=lambda: [])

    def run(self, input) -> Any:
        ...

    # can be passed to wandb.log
    # logging_callback: Callable = logger.info
    # passed to wandb.summary.update
    # summary_callback: Callable = logger.info
    # logging / summary callbacks
    # callbacks: List = [logger.info]
