from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig, OmegaConf
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, r2_score


class Task(Enum):
    regression = 1
    classification = 2


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
    def score(self, X, y):
        ...


class AbstractAdapter(ABC, BaseEstimator):
    _target_: str = MISSING

    @abstractmethod
    def get_data(self) -> Tuple[List, List]:
        ...


class Callback(ABC):
    def __init__(self):
        self.config = None

    def set_config(self, config: Dict):
        self.config = config

    def on_begin(self):
        ...

    def on_config_update(self, config: Dict):
        ...

    def on_log(self, msg: Any, *args: Any):
        ...

    def on_metrics(self, metrics):
        ...

    def on_summary(self, summary: Dict):
        ...

    def on_end(self):
        ...


class AbstractStorageProvider(ABC):
    def __init__(self):
        self.config = None

    def set_config(self, config: Dict):
        self.config = config

    @abstractmethod
    def save(self, filename: str, writer: Callable, mode: str = "w"):
        ...

    @abstractmethod
    def save_pickle(self, filename: str, obj: Any):
        ...

    @abstractmethod
    def restore(self, filename: str, reader: Callable, mode: str = "r") -> Any:
        ...

    @abstractmethod
    def restore_pickle(self, filename: str) -> Any:
        ...
