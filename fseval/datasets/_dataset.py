import logging
import re
from dataclasses import dataclass
from itertools import chain
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from fseval.adapters import Adapter
from fseval.base import Configurable
from fseval.config import DatasetConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@dataclass
class Dataset(DatasetConfig, Configurable):
    # these are "runtime" properties: they are only set once the dataset is loaded.
    n: Optional[int] = None
    p: Optional[int] = None
    multivariate: Optional[bool] = None

    def _get_adapter(self) -> Union[object, tuple]:
        if OmegaConf.is_dict(self.adapter):
            adapter = instantiate(self.adapter)
            return adapter
        elif callable(self.adapter):
            return self.adapter()
        elif isinstance(self.adapter, object):
            adapter = self.adapter
            return adapter
        else:
            raise ValueError(f"Incorrect adapter type: got {type(self.adapter)}.")

    def _get_adapter_data(self) -> Tuple:
        adapter = self._get_adapter()
        if isinstance(adapter, tuple):
            data = adapter
            msg = f"adapter callable `{self._target_}`"
            assert (
                len(data) == 2
            ), f"{msg} must return tuple of length 2 (got {len(data)})."
            X, y = data
            return X, y
        else:
            funcname = self.adapter_callable
            msg = f"adapter class `{self._target_}` function `{funcname}`"
            assert hasattr(adapter, funcname), f"{msg} does not exist."

            get_data_func = getattr(adapter, funcname)
            assert callable(get_data_func), f"{msg} is not callable."

            data = get_data_func()
            assert isinstance(data, tuple), f"{msg} did not return a tuple (X, y)."

            assert (
                len(data) == 2
            ), f"{msg} must return tuple of length 2 (got {len(data)})."
            X, y = data
            return X, y

    def load(self) -> None:
        X, y = self._get_adapter_data()

        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
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
        assert OmegaConf.is_dict(self.feature_importances) or isinstance(
            self.feature_importances, dict
        ), """dataset `feature_importances` ground truth must be a dict."""

        # make variables accessible in current context
        n = self.n
        p = self.p
        X = np.zeros_like(self.X)
        for selector, value in self.feature_importances.items():
            assert (
                re.match(r"X\[.*\]", selector) is not None
            ), f"incorrect feature_importances pattern: {selector} = {value}"

            exec(f"{selector} = {value}")

        # normalize to make every row a probability vector
        row_sums = X.sum(axis=1)
        X = X / row_sums[:, np.newaxis]

        if np.isclose(X, X[0]).all():  # all rows are equal
            return X[0]  # return first row
        else:
            return X  # return entire matrix
