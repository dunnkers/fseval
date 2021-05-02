import logging
from typing import Tuple, List
from dataclasses import dataclass
from fseval.config import DatasetConfig
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Dataset(DatasetConfig):
    def get_data(self) -> Tuple[List, List]:
        raise NotImplementedError

    def load(self) -> None:
        X, y = self.get_data()
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
