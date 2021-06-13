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
        return self.load_dir or "."

    def get_save_dir(self) -> str:
        return self.save_dir or "."

    def save(self, filename: str, writer: Callable, mode: str = "w"):
        filedir = self.get_save_dir()
        filepath = os.path.join(filedir, filename)

        with open(filepath, mode=mode) as file_handle:
            writer(file_handle)

        self.logger.info(
            f"successfully saved {TerminalColor.blue(filename)} to "
            + TerminalColor.yellow("local disk")
            + TerminalColor.green(" ✓")
        )

    def save_pickle(self, filename: str, obj: Any):
        self.save(filename, lambda file: dump(obj, file), mode="wb")

    def restore(self, filename: str, reader: Callable, mode: str = "r") -> Any:
        filedir = self.get_load_dir()
        filepath = os.path.join(filedir, filename)

        if not os.path.exists(filepath):
            return None

        with open(filepath, mode=mode) as file_handle:
            file = reader(file_handle)

        if file:
            self.logger.info(
                f"successfully restored {TerminalColor.blue(filename)} from "
                + TerminalColor.yellow("local disk")
                + TerminalColor.green(" ✓")
            )
            return file
        else:
            return None

    def restore_pickle(self, filename: str) -> Any:
        return self.restore(filename, load, mode="rb")
