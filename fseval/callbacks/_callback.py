import copy
import inspect
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger, getLogger
from typing import Any, Dict, List

from fseval.base import Configurable
from fseval.utils import dict_flatten, dict_merge
from omegaconf import OmegaConf
from yaml import dump

import wandb


class Callback(ABC):
    def __init__(self):
        self.config = None

    def set_config(self, config: Dict):
        self.config = config

    def on_begin(self):
        ...

    def on_config_update(self, config: Dict):
        ...

    def on_log(self, msg: Any, *args: Any):
        ...

    def on_metrics(self, metrics: Dict = None):
        ...

    def on_summary(self, summary: Dict = None):
        ...

    def on_end(self):
        ...


class CallbackList(Callback):
    def __init__(self, callbacks: List = []):
        super(CallbackList, self).__init__()
        self.callbacks = list(callbacks)

    def set_config(self, config: Dict):
        for callback in self.callbacks:
            callback.set_config(config)

    def on_begin(self):
        for callback in self.callbacks:
            callback.on_begin()

    def on_config_update(self, config: Dict):
        for callback in self.callbacks:
            callback.on_config_update(config)

    def on_log(self, msg: Any, *args: Any):
        for callback in self.callbacks:
            callback.on_log(msg, *args)

    def on_metrics(self, metrics: Dict = None):
        for callback in self.callbacks:
            callback.on_metrics(metrics)

    def on_summary(self, summary: Dict = None):
        for callback in self.callbacks:
            callback.on_summary(summary)

    def on_end(self):
        for callback in self.callbacks:
            callback.on_end()
