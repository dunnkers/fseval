import copy
import sys
import time
from dataclasses import dataclass
from logging import Logger, getLogger
from typing import Dict, Optional, cast

import pandas as pd
import wandb
from omegaconf import DictConfig, OmegaConf

from fseval.config.callbacks.to_wandb import ToWandbCallback
from fseval.types import Callback
from fseval.utils.dict_utils import dict_flatten, dict_merge


@dataclass
class WandbCallback(Callback, ToWandbCallback):
    def __post_init__(self):
        if not self.log_metrics:
            logger: Logger = getLogger(__name__)
            logger.warning(
                "logging metrics was disabled by user: "
                + "logging only summary and tables to wandb."
            )

    def _prepare_cfg(self, cfg):
        """Flatten dict and use `/` separators"""
        prepared_cfg = copy.deepcopy(cfg)
        prepared_cfg = dict_flatten(prepared_cfg, sep="/")

        return prepared_cfg

    def on_begin(self, config: DictConfig):
        # convert DictConfig to primitive type
        primitive_cfg = OmegaConf.to_container(config, resolve=True)
        primitive_cfg = cast(Dict, primitive_cfg)
        # prepare
        prepared_cfg = self._prepare_cfg(primitive_cfg)

        # use (1) callback config (2) overriden by pipeline config as input to wandb.init
        init_kwargs = copy.deepcopy(self.wandb_init_kwargs)
        dict_merge(init_kwargs, {"config": prepared_cfg})

        try:
            wandb.init(**init_kwargs)
        except TypeError as e:
            raise type(e)(
                str(e)
                + f""" (make sure whatever config you pass to the wandb callback matches
                the `wandb.init` signature)"""
            ).with_traceback(sys.exc_info()[2])

    def on_config_update(self, config: Dict):
        prepared_cfg = self._prepare_cfg(config)
        wandb.config.update(prepared_cfg, allow_val_change=True)

    def on_metrics(self, metrics: Dict):
        if not self.log_metrics:
            return
        elif isinstance(metrics, Dict):
            wandb.log(metrics)

            # take wandb rate limiting into account: sleep to prevent getting limited
            time.sleep(1.5)
        else:
            raise ValueError(f"Incorrect metric type passed: {type(metrics)}")

    def on_table(self, df: pd.DataFrame, name: str):
        table = wandb.Table(dataframe=df)
        logs = {}
        logs[name] = table
        wandb.log(logs)

    def on_summary(self, summary: Dict):
        wandb.summary.update(summary)

    def on_end(self, exit_code: Optional[int] = None):
        wandb.finish(exit_code=exit_code)
