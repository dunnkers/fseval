import copy
import sys
import time
from collections import UserDict
from logging import Logger, getLogger
from typing import Dict, List, Optional, cast

from fseval.types import Callback
from fseval.utils.dict_utils import dict_flatten, dict_merge
from omegaconf import DictConfig, OmegaConf

import wandb
from wandb.viz import CustomChart, custom_chart_panel_config


class QueryField(dict):
    """Wrapper class for query field panel configs. See `PanelConfig`."""

    def __init__(self, *fields):
        self["name"] = "runSets"
        self["args"] = [{"name": "runSets", "value": r"${runSets}"}]
        self["fields"] = [
            {"name": "id", "fields": []},
            {"name": "name", "fields": []},
            {"name": "_defaultColorIndex", "fields": []},
            *fields,
        ]

    def add_field(self, field: Dict):
        self["fields"].append(field)


class PanelConfig(dict):
    """Wrapper class for panel configs. Is basically what `custom_chart_panel_config`
    in wandb.viz does, but allows more dynamic configuring. Skips the construction
    of a CustomChart class.

    see https://github.com/wandb/client/blob/master/wandb/viz.py"""

    def __init__(self, viz_id: str, fields: Dict = {}, string_fields: Dict = {}):
        self["userQuery"] = {"queryFields": []}
        self["panelDefId"] = viz_id
        self["fieldSettings"] = fields
        self["stringSettings"] = string_fields

    def add_query_field(self, query_field: QueryField):
        self["userQuery"]["queryFields"].append(query_field)


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

    def create_panel_config(
        self,
        viz_id: str,
        table_key: Optional[str] = None,
        summary_fields: List = [],
        config_fields: List = [],
        fields: Dict = {},
        string_fields: Dict = {},
    ) -> PanelConfig:
        """Construct a `PanelConfig`, which is to be passed to the `add_panel_to_run`
        function. Allows uploading custom charts right to the wandb run. Allows
        configuring summary fields, config and tables."""

        # create panel config
        panel_config = PanelConfig(
            viz_id=viz_id, fields=fields, string_fields=string_fields
        )

        ### Add query fields
        query_field = QueryField()

        # add table
        if table_key:
            panel_config["transform"] = {"name": "tableWithLeafColNames"}
            query_field.add_field(
                {
                    "name": "summaryTable",
                    "args": [{"name": "tableKey", "value": table_key}],
                    "fields": [],
                }
            )

        # add summary fields
        if summary_fields:
            query_field.add_field(
                {
                    "name": "summary",
                    "args": [{"name": "keys", "value": summary_fields}],
                    "fields": [],
                }
            )

        # add config fields
        if config_fields:
            query_field.add_field(
                {
                    "name": "config",
                    "args": [{"name": "keys", "value": config_fields}],
                    "fields": [],
                }
            )

        # add query field to panel config
        panel_config.add_query_field(query_field)

        return panel_config

    def add_panel_to_run(
        self, panel_name: str, panel_config: PanelConfig, panel_type: str = "Vega2"
    ) -> None:
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

        # add chart to current run.
        wandb.run._add_panel(panel_name, panel_type, panel_config)

        # "publish" chart to backend
        if wandb.run._backend:
            wandb.run._backend.interface.publish_history({}, wandb.run.step)

    def add_panel(self, panel_name: str, viz_id: str, **panel_config) -> None:
        """Convience method to create panel config, and add a panel to the run with the
        config right away. See `add_panel_to_run()`."""

        panel_config = self.create_panel_config(viz_id, **panel_config)
        self.add_panel_to_run(panel_name, panel_config)
