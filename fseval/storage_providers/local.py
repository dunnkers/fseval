import os
from dataclasses import dataclass
from logging import Logger, getLogger
from pickle import dump, load
from typing import Any, Callable, Optional

from fseval.types import AbstractStorageProvider, TerminalColor


@dataclass
class LocalStorageProvider(AbstractStorageProvider):
    load_dir: Optional[str] = None
    save_dir: Optional[str] = None

    logger: Logger = getLogger(__name__)

    def get_load_dir(self) -> str:
        load_dir = self.load_dir or "."

        return os.path.abspath(load_dir)

    def get_save_dir(self) -> str:
        save_dir = self.save_dir or "."

        return os.path.abspath(save_dir)

    def save(self, filename: str, writer: Callable, mode: str = "w"):
        filedir = self.get_save_dir()
        filepath = os.path.join(filedir, filename)

        with open(filepath, mode=mode) as file_handle:
            writer(file_handle)

        self.logger.debug(
            f"successfully saved {TerminalColor.blue(filename)} to "
            + TerminalColor.yellow("local disk")
            + TerminalColor.green(" ✓")
        )
        self.logger.debug(TerminalColor.blue(filepath))

    def save_pickle(self, filename: str, obj: Any):
        self.save(filename, lambda file: dump(obj, file), mode="wb")

    def restore(self, filename: str, reader: Callable, mode: str = "r") -> Any:
        filedir = self.get_load_dir()
        filepath = os.path.join(filedir, filename)

        self.logger.debug("attempting to restore:")
        self.logger.debug(TerminalColor.blue(filepath))

        if not os.path.exists(filepath):
            return None

        with open(filepath, mode=mode) as file_handle:
            file = reader(file_handle)

        if file:
            self.logger.debug(
                f"successfully restored {TerminalColor.blue(filename)} from "
                + TerminalColor.yellow("local disk")
                + TerminalColor.green(" ✓")
            )
            return file
        else:
            return None

    def restore_pickle(self, filename: str) -> Any:
        return self.restore(filename, load, mode="rb")
