import os
from logging import Logger, getLogger
from pickle import dump, load
from typing import Any, Callable, Dict

import wandb

from fseval.types import AbstractStorageProvider


class WandbStorageProvider(AbstractStorageProvider):
    logger: Logger = getLogger(__name__)

    def set_config(self, config: Dict):
        assert config["callbacks"].get(
            "wandb"
        ), "wandb callback must be enabled to use wandb storage provider."
        super(WandbStorageProvider, self).set_config(config)

    def save(self, filename: str, writer: Callable, mode: str = "w"):
        filedir = wandb.run.dir  # type: ignore
        filepath = os.path.join(filedir, filename)

        with open(filepath, mode=mode) as file_handle:
            writer(file_handle)

        wandb.save(filename, base_path="/")  # type: ignore
        self.logger.info(f"successfully saved `{filename}` to wandb servers ✓")

    def save_pickle(self, filename: str, obj: Any):
        self.save(filename, lambda file: dump(obj, file), mode="wb")

    def _get_restore_file_handle(self, filename: str):
        try:
            file_handle = wandb.restore(filename)
            return file_handle
        except ValueError as err:
            config = self.config
            config_callbacks = config["callbacks"]
            config_wandb = config_callbacks.get("wandb")
            must_resume = config_wandb.get("resume", False) == "must"

            if must_resume:
                self.logger.warn(
                    "wandb callback config got `resume=must` but restoring the file "
                    + f"`{filename}` failed nonetheless:\n"
                    + str(err)  # type: ignore
                )

            return None

    def restore(self, filename: str, reader: Callable, mode: str = "r") -> Any:
        file_handle = self._get_restore_file_handle(filename)

        if not file_handle:
            return None

        filedir = wandb.run.dir  # type: ignore
        filepath = os.path.join(filedir, filename)
        with open(filepath, mode=mode) as file_handle:
            file = reader(file_handle)

        self.logger.info(f"successfully restored `{filename}` from wandb servers ✓")
        return file

    def restore_pickle(self, filename: str) -> Any:
        return self.restore(filename, load, mode="rb")
