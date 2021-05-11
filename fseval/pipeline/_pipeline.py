import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from fseval.base import Configurable
from fseval.config import PipelineConfig
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
    job_type: str
    dataset: Dataset
    cv: CrossValidator
    resample: Resample
    wandb: Dict = field(default_factory=lambda: dict())
    # pipeline component working directory. save files to this dir; wandb will sync them.
    # run_dir: str
    # executes these components in order
    # components: List[Component] = field(default_factory=lambda: [])
    # callbacks
    # callbacks: List[Callback] = field(default_factory=lambda: [])

    def run_pipeline(self, callbacks: List = []):
        callback_list = CallbackList(callbacks)

        callback_list.on_pipeline_begin()
        self.run(None, callback_list)
        callback_list.on_pipeline_end()

    # start wandb right away to capture any errors to its logging system
    # wandb.init(project=cfg.project, config=experiment.get_config())

    # experiment.run()

    # wandb.finish()

    def run(self, input: Any, callback_list: CallbackList) -> Any:
        ...

    # can be passed to wandb.log
    # logging_callback: Callable = logger.info
    # passed to wandb.summary.update
    # summary_callback: Callable = logger.info
    # logging / summary callbacks
    # callbacks: List = [logger.info]
