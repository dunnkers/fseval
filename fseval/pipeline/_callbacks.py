import collections
from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger, getLogger
from typing import Dict, List

import wandb
from fseval.base import Configurable


class Callback(ABC):
    """Abstract class for implementing a pipeline callback.

    Args:
        config: Dict. A copy of the config passed to `pipeline`.
        pipeline: Instance of `fseval.pipeline.Pipeline`. Represents the current
            pipeline being executed.
    """

    def __init__(self):
        self.pipeline = None
        self.config = None

    def set_config(self, config: Dict):
        self.config = config

    def set_pipeline(self, pipeline):
        self.pipeline = pipeline

    def on_pipeline_begin(self, logs: Dict = None):
        ...

    def on_config_update(self, config, logs: Dict = None):
        ...

    def on_log(self, logs: Dict = None):
        ...

    def on_summary(self, logs: Dict = None):
        ...

    def on_pipeline_end(self, logs: Dict = None):
        ...


class CallbackList(Callback):
    def __init__(self, callbacks: List = []):
        super(CallbackList, self).__init__()
        self.callbacks = list(callbacks)

    def set_config(self, config: Dict):
        for callback in self.callbacks:
            callback.set_config(config)

    def set_pipeline(self, pipeline):
        for callback in self.callbacks:
            callback.set_pipeline(pipeline)

    def on_pipeline_begin(self, logs: Dict = None):
        for callback in self.callbacks:
            callback.on_pipeline_begin(logs)

    def on_config_update(self, config, logs: Dict = None):
        for callback in self.callbacks:
            callback.on_config_update(config, logs)

    def on_log(self, logs: Dict = None):
        for callback in self.callbacks:
            callback.on_log(logs)

    def on_summary(self, logs: Dict = None):
        for callback in self.callbacks:
            callback.on_summary(logs)

    def on_pipeline_end(self, logs: Dict = None):
        for callback in self.callbacks:
            callback.on_pipeline_end(logs)


class StdoutCallback(Callback):
    def __init__(self, logger: Logger = getLogger(__name__)):
        super(StdoutCallback, self).__init__()
        self.logger = logger

    def on_pipeline_begin(self, logs: Dict = None):
        self.logger.info("pipeline started.")

    def on_config_update(self, config, logs: Dict = None):
        self.logger.info("config changed: %s", config)

    def on_log(self, logs: Dict = None):
        self.logger.info("received a log: %s", logs)

    def on_summary(self, logs: Dict = None):
        self.logger.info("received a summary: %s", logs)

    def on_pipeline_end(self, logs: Dict = None):
        self.logger.info("pipeline finished.")


class WandbCallback(Callback):
    def on_pipeline_begin(self, logs: Dict = None):
        wandb.init(config=self.config)
        # TODO use `inspect.signature` to populate other kwargs using
        # `config.pop(argname, None)`

        # TODO make sure `job_type` and `group`, etc, are passed on correctly.

    def on_config_update(self, config, logs: Dict = None):
        def dict_merge(dct, merge_dct):
            """Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
            updating only top-level keys, dict_merge recurses down into dicts nested
            to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
            ``dct``.
            :param dct: dict onto which the merge is executed
            :param merge_dct: dct merged into dct
            :return: None
            """
            for k, v in merge_dct.items():
                if (
                    k in dct
                    and isinstance(dct[k], dict)
                    and isinstance(merge_dct[k], collections.Mapping)
                ):
                    dict_merge(dct[k], merge_dct[k])
                else:
                    dct[k] = merge_dct[k]

        # merge recursively, to prevent dangerous overriding operations using
        # `wandb.update`
        dict_merge(wandb.config, config)

    def on_log(self, logs: Dict = None):
        wandb.log(logs)

    def on_summary(self, logs: Dict = None):
        wandb.summary.update(logs)

    def on_pipeline_end(self, logs: Dict = None):
        wandb.finish()
