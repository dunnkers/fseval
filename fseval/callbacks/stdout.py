from logging import Logger, getLogger
from typing import Any, Dict

from fseval.types import Callback
from omegaconf import OmegaConf
from yaml import dump


class StdoutCallback(Callback):
    def __init__(self, logger_name: str = "fseval"):
        super(StdoutCallback, self).__init__()
        self.logger: Logger = getLogger(f"pipeline/{logger_name}:callbacks/stdout")

    def on_begin(self):
        self.logger.info("pipeline started.")

    def on_config_update(self, pipeline_config: Dict):
        yamlized = dump(pipeline_config)
        self.logger.info(f"pipeline config changed: \n{yamlized}")

    def on_log(self, msg: Any, *args: Any):
        self.logger.info(msg, *args)

    def on_end(self):
        self.logger.info("pipeline finished.")
