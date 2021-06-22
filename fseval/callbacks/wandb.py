import copy
import sys
import time
from logging import Logger, getLogger
from typing import Dict, Optional, cast

from fseval.types import Callback
from fseval.utils.dict_utils import dict_flatten, dict_merge
from omegaconf import DictConfig, OmegaConf

import wandb
from wandb.viz import CustomChart, custom_chart_panel_config


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
        init_kwargs = copy.deepcopy(self.callback_config)
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

    def add_panel(
        self,
        viz_id: str,
        panel_name: str,
        table_key: str,
        fields: Dict = {},
        string_fields: Dict = {},
        panel_config_callback=lambda panel_config: panel_config,
    ):
        """Adds a custom chart panel to the current wandb run. This function uses
        internal wandb functions, so might be prone to changes in their code. The
        function is a mixup of the following modules / functions:

        - `wandb.viz`: has `CustomChart` and `custom_chart_panel_config` functions.
            see https://github.com/wandb/client/blob/master/wandb/viz.py
        - `wandb.sdk.wandb_run.Run`: has `_add_panel` and `_backend` functions. This
        function mainly replicates whatever `_history_callback` is doing.
            see https://github.com/wandb/client/blob/master/wandb/sdk/wandb_run.py
        """

        assert wandb.run is not None, "no wandb run in progress. wandb.run is None."

        # create custom chart. is just a data holder class for its attributes.
        custom_chart = CustomChart(
            viz_id=viz_id,
            table=None,
            fields=fields,
            string_fields=string_fields,
        )

        # create custom chart config.
        # Function `custom_chart_panel_config(custom_chart, key, table_key)` has a
        # useless attribute, `key`.
        panel_config = custom_chart_panel_config(custom_chart, None, table_key)
        panel_config = panel_config_callback(panel_config)

        # add chart to current run.
        wandb.run._add_panel(panel_name, "Vega2", panel_config)

        # "publish" chart to backend
        if wandb.run._backend:
            wandb.run._backend.interface.publish_history({}, wandb.run.step)
