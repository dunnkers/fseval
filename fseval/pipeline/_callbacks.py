from abc import ABC
from dataclasses import dataclass
from logging import Logger
from typing import Dict

import wandb
from fseval.base import Configurable


class Callback(ABC):
    def on_pipeline_begin(self, config: Dict = None):
        ...

    def log(self, logs: Dict = None):
        ...

    def summarize(self, logs: Dict = None):
        ...

    def on_pipeline_end(self, config: Dict = None):
        ...


@dataclass
class LoggerCallback(Callback):
    logger: Logger

    def log(self, logs: Dict = None):
        self.logger.info(logs)

    def summarize(self, logs: Dict = None):
        self.logger.info(logs)


class WandbCallback(Callback):
    def on_pipeline_begin(self, config: Dict = None):
        wandb.init(config=config)
        # use `inspect.signature` to populate other kwargs using `config.pop(argname, None)`

    def log(self, logs: Dict = None):
        wandb.log(logs)

    def summarize(self, logs: Dict = None):
        wandb.summary.update(logs)

    def on_pipeline_end(self, config: Dict = None):
        wandb.finish()
