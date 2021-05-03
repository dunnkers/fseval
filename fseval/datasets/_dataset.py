import logging
from typing import Tuple, List, Optional
from dataclasses import dataclass
from fseval.config import DatasetConfig
from fseval.adapters import Adapter
import numpy as np
from sklearn.base import BaseEstimator
from hydra.utils import instantiate

logger = logging.getLogger(__name__)


@dataclass
class Dataset(DatasetConfig, BaseEstimator):
    n: Optional[int] = None
    p: Optional[int] = None
    multivariate: Optional[bool] = None

    def __post_init__(self):
        self.adapter: Adapter = instantiate(self.adapter)

    def load(self) -> None:
        X, y = self.adapter.get_data()
        self.X = np.asarray(X)
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        self.y = np.asarray(y)
        self.multivariate = self.y.ndim > 1
        logger.info(f"loaded {self.name} (n={self.n}, p={self.p})")

    def _ensure_loaded(self) -> None:
        assert hasattr(self, "X"), "please load the dataset first (use `load()`)."

    def get_subsets(self, train_index, test_index) -> Tuple[List, List, List, List]:
        self._ensure_loaded()
        X_train, y_train = self.X[train_index], self.y[train_index]
        X_test, y_test = self.X[test_index], self.y[test_index]
        return X_train, X_test, y_train, y_test
