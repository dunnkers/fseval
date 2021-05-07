import logging
import re
from dataclasses import dataclass
from itertools import chain
from typing import List, Optional, Tuple

import numpy as np
from fseval.adapters import Adapter
from fseval.base import Configurable
from fseval.config import DatasetConfig

logger = logging.getLogger(__name__)


def is_valid_feature_relevancy(selector):
    """Ensures a `feature_relevancy` object is of correct format.

    e.g.
        is_valid("X[:n//2, 0:3]") == True
        is_valid("X[not_allowed, :]") == False
    """
    regex_string = ""
    allowed_chars = [
        "+",  # plus
        "-",  # minus
        "/",  # division. also matches `//`
        "*",  # multiplication. also matches `**`
        "%",  # modulo
        "~",  # negation
        "=",  # equal to
        "[",  # left bracket
        "]",  # right bracket
        "...",  # `...` indexing
        ",",  # comma
        ":",  # `:` slice operator
        "n",  # samples n
        "p",  # dimensions p
    ]
    allowed_chars = [re.escape(s) for s in allowed_chars]
    allowed_chars = "|".join(allowed_chars)

    allowed_patterns = [
        "[0-9]",  # any number
        "\s",  # white space
    ]
    allowed_patterns = "|".join(allowed_patterns)

    allowed = f"{allowed_chars}|{allowed_patterns}"

    match = re.match(f"X\[({allowed})*\]", selector)
    return match


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

    def get_relevancy_selectors(self):
        assert self.feature_relevancy is not None
        selector_list = chain.from_iterable(
            obj.items() for obj in self.feature_relevancy
        )

        for selector, value in selector_list:
            match = is_valid_feature_relevancy(selector)

            assert isinstance(value, float)
            assert match is not None, f"illegal feature relevancy pattern: `{selector}`"

        return selector_list

    @property
    def feature_relevances(self) -> np.ndarray:
        self._ensure_loaded()

        # make variables accessible in current context
        n = self.n
        p = self.p
        X = np.zeros_like(self.X)
        for selector, value in self.get_relevancy_selectors():
            exec(f"{selector} = {value}")
        return X
