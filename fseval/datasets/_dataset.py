import logging
import re
from dataclasses import dataclass
from itertools import chain
from typing import Dict, List, Optional, Tuple

import numpy as np
from fseval.adapters import Adapter
from fseval.base import Configurable
from fseval.config import DatasetConfig
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@dataclass
class Dataset(DatasetConfig, Configurable):
    n: Optional[int] = None
    p: Optional[int] = None
    multivariate: Optional[bool] = None

    def load(self) -> None:
        X, y = self.adapter.get_data()
        self.X = np.asarray(X)
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        self.y = np.asarray(y)
        self.multivariate = self.y.ndim > 1
        task_name = self.task.name if hasattr(self.task, "name") else self.task
        logger.info(f"loaded {self.name} {task_name} dataset (n={self.n}, p={self.p})")

    def _ensure_loaded(self) -> None:
        assert hasattr(self, "X"), "please load the dataset first (use `load()`)."

    def get_subsets(self, train_index, test_index) -> Tuple[List, List, List, List]:
        self._ensure_loaded()
        X_train, y_train = self.X[train_index], self.y[train_index]
        X_test, y_test = self.X[test_index], self.y[test_index]
        return X_train, X_test, y_train, y_test

    def get_feature_importances(self) -> Optional[np.ndarray]:
        self._ensure_loaded()

        if self.feature_importances is None:
            return None
        assert isinstance(
            self.feature_importances, DictConfig
        ), """dataset `feature_importances` ground truth has incorrect format: must be 
        a dictionary of type `Dict[str, float]`."""

        # make variables accessible in current context
        n = self.n
        p = self.p
        X = np.zeros_like(self.X)
        for selector, value in self.feature_importances.items():
            assert (
                re.match("X\[.*\]", selector) is not None
            ), f"incorrect feature_importances pattern: {selector} = {value}"

            exec(f"{selector} = {value}")

        # normalize to make every row a probability vector
        row_sums = X.sum(axis=1)
        X /= row_sums[:, np.newaxis]

        if np.isclose(X, X[0]).all():  # all rows are equal
            return X[0]  # return first row
        else:
            return X  # return entire matrix
