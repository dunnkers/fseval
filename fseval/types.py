from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.base import BaseEstimator


class Task(Enum):
    """Learning task. In the case of datasets this indicates the dataset learning task,
    and in the case of estimators this indicates the supported estimator learning tasks.
    """

    regression = 1
    classification = 2


class CacheUsage(Enum):
    """
    Determines how cache usage is handled. In the case of **loading** caches:

    - `allow`: program might use cache; if found and could be restored
    - `must`: program should fail if no cache found
    - `never`: program should not load cache even if found

    When **saving** caches:
    - `allow`: program might save cache; no fatal error thrown when fails
    - `must`: program must save cache; throws error if fails (e.g. due to out of memory)
    - `never`: program does not try to save a cached version
    """

    allow = 1
    must = 2
    never = 3


class IncompatibilityError(Exception):
    ...


class AbstractEstimator(ABC, BaseEstimator):
    @abstractmethod
    def fit(self, X, y):
        ...

    @abstractmethod
    def score(self, X, y, **kwargs) -> Union[Dict, pd.DataFrame, np.generic, None]:
        ...


class AbstractAdapter(ABC, BaseEstimator):
    @abstractmethod
    def get_data(self) -> Tuple[List, List]:
        ...


class Callback(ABC):
    def on_begin(self, config: DictConfig):
        ...

    def on_config_update(self, config: Dict):
        ...

    def on_metrics(self, metrics):
        ...

    def on_table(self, df: pd.DataFrame, name: str):
        ...

    def on_summary(self, summary: Dict):
        ...

    def on_end(self, exit_code: Optional[int] = None):
        ...


class AbstractStorage(ABC):
    @abstractmethod
    def get_load_dir(self) -> str:
        ...

    @abstractmethod
    def get_save_dir(self) -> str:
        ...

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


class AbstractPipeline(AbstractEstimator, ABC):
    @abstractmethod
    def prefit(self):
        ...

    @abstractmethod
    def postfit(self):
        ...


class TerminalColor:
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


class AbstractMetric:
    def score_bootstrap(
        self,
        ranker: AbstractEstimator,
        validator: AbstractEstimator,
        callbacks: Callback,
        scores: Dict,
        **kwargs,
    ) -> Dict:
        """Aggregated metrics for the bootstrapped pipeline."""
        ...

    def score_pipeline(self, scores: Dict, callbacks: Callback, **kwargs) -> Dict:
        """Aggregated metrics for the pipeline."""
        ...

    def score_ranking(
        self,
        scores: Union[Dict, pd.DataFrame],
        ranker: AbstractEstimator,
        bootstrap_state: int,
        callbacks: Callback,
        feature_importances: Optional[np.ndarray] = None,
    ) -> Union[Dict, pd.DataFrame]:
        """Metrics for validating a feature ranking, e.g. using a ground-truth."""
        ...

    def score_support(
        self,
        scores: Union[Dict, pd.DataFrame],
        validator: AbstractEstimator,
        X,
        y,
        callbacks: Callback,
        **kwargs,
    ) -> Union[Dict, pd.DataFrame]:
        """Metrics for validating a feature support vector. e.g., this is an array
        indicating yes/no which features to include in a feature subset. The array is
        validated by running the validation estimator on this feature subset."""
        ...

    def score_dataset(
        self, scores: Union[Dict, pd.DataFrame], callbacks: Callback, **kwargs
    ) -> Union[Dict, pd.DataFrame]:
        """Aggregated metrics for all feature subsets. e.g. 50 feature subsets for
        p >= 50."""
        ...

    def score_subset(
        self,
        scores: Union[Dict, pd.DataFrame],
        validator: AbstractEstimator,
        X,
        y,
        callbacks: Callback,
        **kwargs,
    ) -> Union[Dict, pd.DataFrame]:
        """Metrics for validation estimator. Validates 1 feature subset."""
        ...
