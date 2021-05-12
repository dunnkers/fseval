import copy
import inspect
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger, getLogger
from typing import Dict, List

import wandb
from fseval.base import Configurable
from fseval.utils import dict_merge
from omegaconf import OmegaConf


class Callback(ABC):
    """Abstract class for implementing a pipeline callback.

    Args:
        pipeline_config: Dict. A copy of the config passed to `pipeline`.
        pipeline: Instance of `fseval.pipeline.Pipeline`. Represents the current
            pipeline being executed.
    """

    def __init__(self):
        self.pipeline_config = None
        self.pipeline = None

    def set_pipeline_config(self, pipeline_config: Dict):
        self.pipeline_config = pipeline_config

    def set_pipeline(self, pipeline):
        self.pipeline = pipeline

    def on_pipeline_begin(self, logs: Dict = None):
        ...

    def on_pipeline_config_update(self, pipeline_config: Dict, logs: Dict = None):
        ...

    def on_log(self, logs: Dict = None):
        ...

    def on_summary(self, logs: Dict = None):
        ...

    def on_file_save(self, filename, content):
        ...

    def on_pipeline_end(self, logs: Dict = None):
        ...


class CallbackList(Callback):
    def __init__(self, callbacks: List = []):
        super(CallbackList, self).__init__()
        self.callbacks = list(callbacks)

    def set_pipeline_config(self, pipeline_config: Dict):
        for callback in self.callbacks:
            callback.set_pipeline_config(pipeline_config)

    def set_pipeline(self, pipeline):
        for callback in self.callbacks:
            callback.set_pipeline(pipeline)

    def on_pipeline_begin(self, logs: Dict = None):
        for callback in self.callbacks:
            callback.on_pipeline_begin(logs)

    def on_pipeline_config_update(self, pipeline_config: Dict, logs: Dict = None):
        for callback in self.callbacks:
            callback.on_pipeline_config_update(pipeline_config, logs)

    def on_log(self, logs: Dict = None):
        for callback in self.callbacks:
            callback.on_log(logs)

    def on_summary(self, logs: Dict = None):
        for callback in self.callbacks:
            callback.on_summary(logs)

    def on_file_save(self, filename, content):
        for callback in self.callbacks:
            callback.on_file_save(filename, content)

    def on_pipeline_end(self, logs: Dict = None):
        for callback in self.callbacks:
            callback.on_pipeline_end(logs)


class StdoutCallback(Callback):
    def __init__(self, logger: Logger = getLogger(__name__)):
        super(StdoutCallback, self).__init__()
        self.logger = logger

    def on_pipeline_begin(self, logs: Dict = None):
        self.logger.info("pipeline started.")

    def on_pipeline_config_update(self, pipeline_config: Dict, logs: Dict = None):
        self.logger.info("pipeline config changed: %s", pipeline_config)

    def on_log(self, logs: Dict = None):
        self.logger.info("received a log: %s", logs)

    def on_summary(self, logs: Dict = None):
        self.logger.info("received a summary: %s", logs)

    def on_file_save(self, filename, content):
        self.logger.info("received a file: %s", filename)

    def on_pipeline_end(self, logs: Dict = None):
        self.logger.info("pipeline finished.")


class WandbCallback(Callback):
    def __init__(self, **kwargs):
        super(WandbCallback, self).__init__()
        # make sure any nested objects are casted from DictConfig's to regular dict's.
        kwargs = OmegaConf.create(kwargs)
        kwargs = OmegaConf.to_container(kwargs)

        self.callback_config = kwargs

    def on_pipeline_begin(self, logs: Dict = None):
        # use (1) callback config (2) overriden by pipeline config as input to wandb.init
        init_kwargs = copy.deepcopy(self.callback_config)
        pipeline_config = copy.deepcopy(self.pipeline_config)
        dict_merge(init_kwargs, {"config": pipeline_config})

        try:
            wandb.init(**init_kwargs)
        except TypeError as e:
            raise type(e)(
                str(e)
                + f""" (make sure whatever config you pass to the wandb callback matches
                the `wandb.init` signature)"""
            ).with_traceback(sys.exc_info()[2])

    def on_pipeline_config_update(self, pipeline_config: Dict, logs: Dict = None):
        # merge recursively, to prevent overriding pipeline config using `wandb.update`
        dict_merge(wandb.config, pipeline_config)

    def on_log(self, logs: Dict = None):
        wandb.log(logs)

    def on_summary(self, logs: Dict = None):
        wandb.summary.update(logs)

    def on_file_save(self, filename, content):
        filepath = os.path.join(wandb.run.dir, filename)
        f = open(filepath, "w")
        f.write(content)
        f.close()

        wandb.save(filename, base_path="/")

    def on_pipeline_end(self, logs: Dict = None):
        wandb.finish()
