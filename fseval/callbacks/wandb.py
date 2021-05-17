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


class WandbCallback(Callback):
    def __init__(self, **kwargs):
        super(WandbCallback, self).__init__()
        # make sure any nested objects are casted from DictConfig's to regular dict's.
        kwargs = OmegaConf.create(kwargs)
        kwargs = OmegaConf.to_container(kwargs)

        self.callback_config = kwargs

    def on_begin(self):
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

    def on_config_update(self, pipeline_config: Dict):
        # merge recursively, to prevent overriding pipeline config using `wandb.update`
        dict_merge(wandb.config, pipeline_config)

    def on_metrics(self, metrics: Dict = None):
        wandb.log(metrics)

    def on_summary(self, summary: Dict = None):
        wandb.summary.update(summary)

    def on_end(self):
        wandb.finish()
