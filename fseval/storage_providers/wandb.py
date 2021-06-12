import os
from dataclasses import dataclass
from logging import Logger, getLogger
from pickle import dump, load
from typing import Any, Callable, Dict, Optional

from fseval.types import AbstractStorageProvider, TerminalColor
from omegaconf import DictConfig

import wandb

from .local import LocalStorageProvider


@dataclass
class WandbStorageProvider(LocalStorageProvider):
    """Storage provider for Weights and Biases (wandb), allowing users to save- and
    restore files to the service.

    Arguments:
        local_dir: Optional[str] - when set, an attempt is made to load from the
        designated local directory first, before downloading the data off of wandb. Can
        be used to perform faster loads or prevent being rate-limited on wandb.

        entity: Optional[str] - allows you to recover from a specific entity,
        instead of using the entity that is set for the 'current' run.

        project: Optional[str] - recover from a specific project.

        run_id: Optional[str] - recover from a specific run id."""

    entity: Optional[str] = None
    project: Optional[str] = None
    run_id: Optional[str] = None

    logger: Logger = getLogger(__name__)

    def _assert_wandb_available(self):
        assert wandb.run is not None, (
            "`wandb.run` is not available in this process. you are perhaps using multi-"
            + "processing: make sure to only use the wandb storage provider from the main"
            + "thread. see https://docs.wandb.ai/guides/track/advanced/distributed-training."
        )

    # TODO better error when callback not enabled, i.e.
    # assert config["callbacks"].get(
    #     "wandb"
    # ), "wandb callback must be enabled to use wandb storage provider."

    def _get_local_dir(self) -> str:
        self._assert_wandb_available()
        filedir = wandb.run.dir  # type: ignore
        return filedir

    # a=wandb.Api().run("dunnkers/uncategorized/1wthczn7").config
    def save(self, filename: str, writer: Callable, mode: str = "w"):
        # save to local disk
        super(WandbStorageProvider, self).save(filename, writer, mode)

        # save to wandb
        wandb.save(filename, base_path="/")  # type: ignore
        self.logger.info(
            f"uploaded {TerminalColor.yellow(filename)} to "
            + TerminalColor.blue("wandb servers")
            + TerminalColor.green(" ✓")
        )

    def _restore_from_wandb(self, filename: str):
        try:
            entity = self.entity or wandb.run.entity  # type: ignore
            project = self.project or wandb.run.project  # type: ignore
            run_id = self.run_id or wandb.run.id  # type: ignore
            run_path = f"{entity}/{project}/{run_id}"
            file_handle = wandb.restore(filename, run_path=run_path)

            return file_handle
        except ValueError as err:
            return None

    def restore(self, filename: str, reader: Callable, mode: str = "r") -> Any:
        """Given a filename, restores the file either from local disk or from wandb,
        depending on the availability of the file. First, the local disk is searched
        for the file, taking in regard the `local_dir` value in the
        `WandbStorageProvider` constructor. If this file is not found, the file will
        be downloaded fresh from wandb servers."""

        # (1) attempt local restoration if available
        file = super(WandbStorageProvider, self).restore(filename, reader, mode)
        if file:
            return file

        # (2) otherwise, restore by downloading from wandb
        file = self._restore_from_wandb(filename)
        if file:
            self.logger.info(
                f"downloaded {TerminalColor.yellow(filename)} from "
                + TerminalColor.blue("wandb servers")
                + TerminalColor.green(" ✓")
            )
            file = super(WandbStorageProvider, self).restore(filename, reader, mode)
            return file

        # (3) if no cache is available anywhere, return None.
        return None
