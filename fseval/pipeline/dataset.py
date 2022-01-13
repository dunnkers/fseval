import re
from dataclasses import dataclass
from logging import Logger, getLogger
from typing import Optional, Tuple, Union

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from fseval.config import DatasetConfig
from fseval.types import TerminalColor


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    n: int
    p: int
    multioutput: bool
    feature_importances: Optional[np.ndarray] = None

    logger: Logger = getLogger(__name__)

    @property
    def _log_details(self):
        details = []
        details.append(f"n={self.n}")
        details.append(f"p={self.p}")

        if self.multioutput:
            details.append("multioutput")

        details = [TerminalColor.yellow(detail) for detail in details]
        details_str = ",".join(details)
        return details_str


@dataclass
class DatasetLoader(DatasetConfig):
    logger: Logger = getLogger(__name__)

    def _get_adapter(self) -> Union[object, tuple]:
        if OmegaConf.is_dict(self.adapter):
            adapter = instantiate(self.adapter)
            return adapter
        elif callable(self.adapter):
            return self.adapter()
        else:
            return self.adapter

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

    def get_feature_importances(
        self, X: np.ndarray, n: int, p: int
    ) -> Optional[np.ndarray]:
        if self.feature_importances is None:
            return None

        assert OmegaConf.is_dict(self.feature_importances) or isinstance(
            self.feature_importances, dict
        ), """dataset `feature_importances` ground truth must be a dict."""

        # make variables accessible in current context
        X = np.zeros_like(X)

        # process feature importances
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

    def load(self) -> Dataset:
        self.logger.info(f"task: {self.task.name}")
        self.logger.info(f"loading dataset {TerminalColor.yellow(self.name)}...")
        X, y = self._get_adapter_data()

        X = np.asarray(X)
        y = np.asarray(y)
        n = X.shape[0]
        p = X.shape[1]
        multioutput = y.ndim > 1
        feature_importances = self.get_feature_importances(X, n, p)

        dataset = Dataset(self.name, X, y, n, p, multioutput, feature_importances)
        self.logger.info(
            f"loaded dataset {TerminalColor.yellow(self.name)} "
            + TerminalColor.green("✓")
        )
        return dataset
