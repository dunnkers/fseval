import os
from dataclasses import dataclass
from logging import Logger, getLogger
from pickle import dump, load
from typing import Any, Callable, Dict, Optional

from fseval.types import AbstractStorageProvider, TerminalColor

import wandb


@dataclass
class WandbStorageProvider(AbstractStorageProvider):
    """Storage provider for Weights and Biases (wandb), allowing users to save- and
    restore files to the service.

    Arguments:
        local_dir: Optional[str] - when set, an attempt is made to load from the
        designated local directory first, before downloading the data off of wandb. Can
        be used to perform faster loads or prevent being rate-limited on wandb.

        wandb_entity: Optional[str] - allows you to recover from a specific entity,
        instead of using the entity that is set for the 'current' run.
        wandb_project: Optional[str] - idem
        wandb_run_id: Optional[str] - idem"""

    local_dir: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_run_id: Optional[str] = None
    logger: Logger = getLogger(__name__)

    def _assert_wandb_available(self):
        assert wandb.run is not None, (
            "`wandb.run` is not available in this process. you are perhaps using multi-"
            + "processing: make sure to only use the wandb storage provider from the main"
            + "thread. see https://docs.wandb.ai/guides/track/advanced/distributed-training."
        )

    def set_config(self, config: Dict):
        assert config["callbacks"].get(
            "wandb"
        ), "wandb callback must be enabled to use wandb storage provider."
        super(WandbStorageProvider, self).set_config(config)

    def save(self, filename: str, writer: Callable, mode: str = "w"):
        self._assert_wandb_available()

        filedir = wandb.run.dir  # type: ignore
        filepath = os.path.join(filedir, filename)

        with open(filepath, mode=mode) as file_handle:
            writer(file_handle)

        wandb.save(filename, base_path="/")  # type: ignore
        self.logger.info(
            f"successfully saved {TerminalColor.yellow(filename)} to wandb servers "
            + TerminalColor.green("✓")
        )

    def save_pickle(self, filename: str, obj: Any):
        self.save(filename, lambda file: dump(obj, file), mode="wb")

    def _get_restore_file_handle(self, filename: str):
        try:
            entity = self.wandb_entity or wandb.run.entity  # type: ignore
            project = self.wandb_project or wandb.run.project  # type: ignore
            run_id = self.wandb_run_id or wandb.run.id  # type: ignore

            file_handle = wandb.restore(
                filename, run_path=f"{entity}/{project}/{run_id}"
            )
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

    def _local_restoration(self, filename: str, reader: Callable, mode: str = "r"):
        local_file = os.path.join(self.local_dir or "", filename)
        if self.local_dir is not None and os.path.exists(local_file):
            filepath = local_file

            with open(filepath, mode=mode) as file_handle:
                file = reader(file_handle)

            return file or None
        else:
            return None

    def _wandb_restoration(self, filename: str, reader: Callable, mode: str = "r"):
        if self._get_restore_file_handle(filename):
            filedir = wandb.run.dir  # type: ignore
            filepath = os.path.join(filedir, filename)

            with open(filepath, mode=mode) as file_handle:
                file = reader(file_handle)

            return file or None
        else:
            return None

    def restore(self, filename: str, reader: Callable, mode: str = "r") -> Any:
        """Given a filename, restores the file either from local disk or from wandb,
        depending on the availability of the file. First, the local disk is searched
        for the file, taking in regard the `local_dir` value in the
        `WandbStorageProvider` constructor. If this file is not found, the file will
        be downloaded fresh from wandb servers."""

        self._assert_wandb_available()

        # (1) attempt local restoration if available
        file = self._local_restoration(filename, reader, mode)
        if file:
            self.logger.info(
                f"successfully restored {TerminalColor.yellow(filename)} from "
                + TerminalColor.blue("disk ")
                + TerminalColor.green("✓")
            )

            return file

        # (2) otherwise, restore by downloading from wandb
        file = self._wandb_restoration(filename, reader, mode)
        if file:
            self.logger.info(
                f"successfully restored {TerminalColor.yellow(filename)} from "
                + TerminalColor.blue("wandb servers ")
                + TerminalColor.green("✓")
            )

            return file

        # (3) if no cache is available anywhere, return None.
        return None

    def restore_pickle(self, filename: str) -> Any:
        return self.restore(filename, load, mode="rb")
