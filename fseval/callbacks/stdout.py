import copy
import inspect
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger, getLogger
from typing import Any, Dict, List

from fseval.utils import dict_flatten, dict_merge
from omegaconf import OmegaConf
from yaml import dump

import wandb

from ._callback import Callback


class StdoutCallback(Callback):
    def __init__(self, logger_name: str = "fseval"):
        super(StdoutCallback, self).__init__()
        self.logger: Logger = getLogger(f"pipeline/{logger_name}:callbacks/stdout")

    def on_begin(self):
        self.logger.info("pipeline started.")

    def on_config_update(self, pipeline_config: Dict):
        flattend = dict_flatten(pipeline_config)
        yamlized = dump(flattend)
        self.logger.info(f"pipeline config changed: \n{yamlized}")

    def on_log(self, msg: Any, *args: Any):
        self.logger.info(msg, *args)

    def on_end(self):
        self.logger.info("pipeline finished.")
