from typing import Any, Dict, List

from omegaconf import OmegaConf
from yaml import dump

import wandb
from fseval.types import Callback


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

    def on_metrics(self, metrics):
        for callback in self.callbacks:
            callback.on_metrics(metrics)

    def on_summary(self, summary):
        for callback in self.callbacks:
            callback.on_summary(summary)

    def on_end(self):
        for callback in self.callbacks:
            callback.on_end()
