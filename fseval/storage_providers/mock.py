from typing import Any, Callable

from fseval.types import AbstractStorageProvider


class MockStorageProvider(AbstractStorageProvider):
    def save(self, filename: str, writer: Callable, mode: str = "w"):
        ...

    def save_pickle(self, filename: str, obj: Any):
        ...

    def restore(self, filename: str, reader: Callable, mode: str = "r") -> Any:
        ...

    def restore_pickle(self, filename: str) -> Any:
        ...
