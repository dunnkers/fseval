import copy
import sys
import time
from logging import Logger, getLogger
from typing import Dict, Optional

from fseval.types import Callback
from omegaconf import OmegaConf

import wandb


# Recursive dictionary merge
# Copyright (C) 2016 Paul Durivage <pauldurivage+github@gmail.com>
# https://gist.github.com/angstwad/bf22d1822c38a92ec0a9
def dict_merge(dct, merge_dct):
    """Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.

    Args:
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


class WandbCallback(Callback):
    def __init__(self, **kwargs):
        super(WandbCallback, self).__init__()
        # make sure any nested objects are casted from DictConfig's to regular dict's.
        kwargs = OmegaConf.create(kwargs)
        kwargs = OmegaConf.to_container(kwargs)

        # whether to log metrics. in the case of a resumation run, a user might probably
        # not want to log metrics, but just update the tables instead.
        kwargs.setdefault("log_metrics", True)
        self.log_metrics = kwargs.pop("log_metrics")

        if not self.log_metrics:
            logger: Logger = getLogger(__name__)
            logger.warn(
                "logging metrics was disabled by user: "
                + "logging only summary and tables to wandb."
            )

        # the `kwargs` are passed to `wandb.init`
        self.callback_config = kwargs

    def on_begin(self):
        # use (1) callback config (2) overriden by pipeline config as input to wandb.init
        init_kwargs = copy.deepcopy(self.callback_config)
        config = copy.deepcopy(self.config)
        dict_merge(init_kwargs, {"config": config})

        try:
            wandb.init(**init_kwargs)
        except TypeError as e:
            raise type(e)(
                str(e)
                + f""" (make sure whatever config you pass to the wandb callback matches
                the `wandb.init` signature)"""
            ).with_traceback(sys.exc_info()[2])

    def on_config_update(self, config: Dict):
        # merge recursively, to prevent overriding pipeline config using `wandb.update`
        dict_merge(wandb.config, config)

    def on_metrics(self, metrics):
        if not self.log_metrics:
            return
        elif isinstance(metrics, Dict):
            wandb.log(metrics)

            # take wandb rate limiting into account: sleep to prevent getting limited
            time.sleep(1.5)
        else:
            raise ValueError(f"Incorrect metric type passed: {type(metrics)}")

    def on_summary(self, summary: Dict):
        wandb.summary.update(summary)

    def on_end(self, exit_code: Optional[int] = None):
        wandb.finish(exit_code=exit_code)

    def upload_table_plot(self, df, **kwargs):
        table = wandb.Table(dataframe=df)
        return wandb.plot_table(data_table=table, **kwargs)

    def upload_table(self, df, name):
        table = wandb.Table(dataframe=df)
        logs = {}
        logs[name] = table
        wandb.log(logs)
