from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from omegaconf import MISSING, DictConfig
from sklearn.base import BaseEstimator


class Task(Enum):
    regression = 1
    classification = 2


class IncompatibilityError(Exception):
    ...


class AbstractEstimator(ABC, BaseEstimator):
    @abstractmethod
    def fit(self, X, y):
        ...

    @abstractmethod
    def transform(self, X, y):
        ...

    @abstractmethod
    def fit_transform(self, X, y):
        ...

    @abstractmethod
    def score(self, X, y, **kwargs):
        ...


class AbstractAdapter(ABC, BaseEstimator):
    _target_: str = MISSING

    @abstractmethod
    def get_data(self) -> Tuple[List, List]:
        ...


class Callback(ABC):
    def on_begin(self, config: DictConfig):
        ...

    def on_config_update(self, config: Dict):
        ...

    def on_log(self, msg: Any, *args: Any):
        ...

    def on_metrics(self, metrics):
        ...

    def on_summary(self, summary: Dict):
        ...

    def on_end(self, exit_code: Optional[int] = None):
        ...


class AbstractStorageProvider(ABC):
    def save(self, filename: str, writer: Callable, mode: str = "w"):
        ...

    def save_pickle(self, filename: str, obj: Any):
        ...

    def restore(self, filename: str, reader: Callable, mode: str = "r") -> Any:
        ...

    def restore_pickle(self, filename: str) -> Any:
        ...


class AbstractPipeline(AbstractEstimator, ABC):
    @abstractmethod
    def prefit(self):
        ...

    @abstractmethod
    def postfit(self):
        ...


class TerminalColor:
    @staticmethod
    def black(text):
        return f"\u001b[30m{text}\u001b[0m"

    @staticmethod
    def red(text):
        return f"\u001b[31m{text}\u001b[0m"

    @staticmethod
    def green(text):
        return f"\u001b[32m{text}\u001b[0m"

    @staticmethod
    def yellow(text):
        return f"\u001b[33m{text}\u001b[0m"

    @staticmethod
    def blue(text):
        return f"\u001b[34m{text}\u001b[0m"

    @staticmethod
    def purple(text):
        return f"\u001b[35m{text}\u001b[0m"

    @staticmethod
    def cyan(text):
        return f"\u001b[36m{text}\u001b[0m"

    @staticmethod
    def grey(text):
        return f"\u001b[37m{text}\u001b[0m"
