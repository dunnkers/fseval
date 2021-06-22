from dataclasses import dataclass
from logging import Logger, getLogger
from os import path
from typing import Any, Callable, Optional

from fseval.types import TerminalColor

import wandb

from .local import LocalStorageProvider


@dataclass
class WandbStorageProvider(LocalStorageProvider):
    """Storage provider for Weights and Biases (wandb), allowing users to save- and
    restore files to the service.

    Arguments:
        load_dir: Optional[str] - when set, an attempt is made to load from the
        designated local directory first, before downloading the data off of wandb. Can
        be used to perform faster loads or prevent being rate-limited on wandb.

        save_dir: Optional[str] - when set, uses this directory to save files, instead
        of the usual wandb run directory, under the `files` subdirectory.

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
            "`wandb.run` is not available in this process. this can be because either: "
            + "(1) the wandb callback is not enabled. enable it by setting "
            + "`callbacks=[wandb]`. "
            + "(2) you are using multi-processing: make sure to only use the wandb "
            + "storage provider from the main thread. "
            + "see https://docs.wandb.ai/guides/track/advanced/distributed-training."
        )

    def _get_wandb_run_path(self) -> str:
        entity: str = self.entity or wandb.run.entity  # type: ignore
        project: str = self.project or wandb.run.project  # type: ignore
        run_id: str = self.run_id or wandb.run.id  # type: ignore
        run_path = f"{entity}/{project}/{run_id}"

        return run_path

    def _get_wandb_load_dir(self) -> str:
        self._assert_wandb_available()

        # (1) run was resumed: use last run's local dir.
        if wandb.run.resumed:  # type: ignore
            load_dir = wandb.run.config["storage_provider/save_dir"]  # type: ignore
            self.logger.info(
                f"{TerminalColor.yellow('loading')} files from: "
                + TerminalColor.cyan("wandb save dir")
                + " (run was resumed)"
            )

            return load_dir

        # (2) use configured run, or otherwise current directory.
        run_path = self._get_wandb_run_path()
        api = wandb.Api()
        try:
            run = api.run(run_path)
            load_dir = run.config["storage_provider/save_dir"]
            self.logger.info(
                f"{TerminalColor.yellow('loading')} files from: "
                + TerminalColor.cyan("remote run")
            )

            return load_dir
        except Exception:
            ...

        # (3) use current directory.
        load_dir = wandb.run.dir  # type: ignore
        self.logger.info(
            f"{TerminalColor.yellow('loading')} files from: "
            + TerminalColor.cyan("current directory")
            + " (no existing run found)"
        )

        return load_dir

    def get_load_dir(self) -> str:
        load_dir = self.load_dir or self._get_wandb_load_dir()
        self.load_dir = load_dir

        return path.abspath(load_dir)

    def _get_wandb_save_dir(self) -> str:
        self._assert_wandb_available()

        # for saving, always use the current run dir.
        save_dir = wandb.run.dir  # type: ignore
        self.logger.info(
            TerminalColor.yellow("saving")
            + " files to: "
            + TerminalColor.cyan("wandb run dir")
        )

        return save_dir

    def get_save_dir(self) -> str:
        save_dir = self.save_dir or self._get_wandb_save_dir()
        self.save_dir = save_dir

        return path.abspath(save_dir)

    def save(self, filename: str, writer: Callable, mode: str = "w"):
        # save to local disk
        super(WandbStorageProvider, self).save(filename, writer, mode)

        # save to wandb
        wandb.save(filename, base_path="/")  # type: ignore
        self.logger.info(
            f"uploaded {TerminalColor.blue(filename)} to "
            + TerminalColor.yellow("wandb servers")
            + TerminalColor.green(" ✓")
        )

    def _restore_from_wandb(self, filename: str):
        try:
            run_path = self._get_wandb_run_path()
            file_handle = wandb.restore(filename, run_path=run_path)

            return file_handle
        except ValueError:
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
                f"downloaded {TerminalColor.blue(filename)} from "
                + TerminalColor.yellow("wandb servers")
                + TerminalColor.green(" ✓")
            )
            file = super(WandbStorageProvider, self).restore(filename, reader, mode)
            return file

        # (3) if no cache is available anywhere, return None.
        return None
