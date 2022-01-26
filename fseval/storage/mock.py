from dataclasses import dataclass
from typing import Any, Callable

from fseval.config.storage import MockStorageConfig
from fseval.types import AbstractStorage


@dataclass
class MockStorage(AbstractStorage, MockStorageConfig):
    def get_load_dir(self) -> str:
        ...

    def get_save_dir(self) -> str:
        ...

    def save(self, filename: str, writer: Callable, mode: str = "w"):
        ...

    def save_pickle(self, filename: str, obj: Any):
        ...

    def restore(self, filename: str, reader: Callable, mode: str = "r") -> Any:
        ...

    def restore_pickle(self, filename: str) -> Any:
        ...
