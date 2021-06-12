import os
from dataclasses import dataclass
from logging import Logger, getLogger
from pickle import dump, load
from typing import Any, Callable, Dict, Optional

from fseval.types import AbstractStorageProvider, TerminalColor
from omegaconf import DictConfig


@dataclass
class LocalStorageProvider(AbstractStorageProvider):
    local_dir: Optional[str] = None

    logger: Logger = getLogger(__name__)

    def _get_local_dir(self) -> str:
        return self.local_dir or "."

    def save(self, filename: str, writer: Callable, mode: str = "w"):
        filedir = self._get_local_dir()
        filepath = os.path.join(filedir, filename)

        with open(filepath, mode=mode) as file_handle:
            writer(file_handle)

        self.logger.info(
            f"successfully saved {TerminalColor.yellow(filename)} to "
            + TerminalColor.blue("local disk")
            + TerminalColor.green(" ✓")
        )

    def save_pickle(self, filename: str, obj: Any):
        self.save(filename, lambda file: dump(obj, file), mode="wb")

    def restore(self, filename: str, reader: Callable, mode: str = "r") -> Any:
        filedir = self._get_local_dir()
        filepath = os.path.join(filedir, filename)

        if self.local_dir is None or not os.path.exists(filepath):
            return None

        with open(filepath, mode=mode) as file_handle:
            file = reader(file_handle)

        if file:
            self.logger.info(
                f"successfully restored {TerminalColor.yellow(filename)} from "
                + TerminalColor.blue("local disk")
                + TerminalColor.green(" ✓")
            )
        else:
            return None

    def restore_pickle(self, filename: str) -> Any:
        return self.restore(filename, load, mode="rb")
