from dataclasses import dataclass
from typing import Any, Callable, Optional

from fseval.types import AbstractStorage


@dataclass
class MockStorage(AbstractStorage):
    load_dir: Optional[str] = None
    save_dir: Optional[str] = None

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