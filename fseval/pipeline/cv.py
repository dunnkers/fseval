import logging
from dataclasses import dataclass
from typing import Any, Generator, List, Tuple

from omegaconf import MISSING

logger = logging.getLogger(__name__)


@dataclass
class CrossValidatorConfig:
    """
    Parameters of both BaseCrossValidator and BaseShuffleSplit.
    """

    _target_: str = "fseval.pipeline.cv.CrossValidator"
    name: str = MISSING
    """ splitter. must be BaseCrossValidator or BaseShuffleSplit; should at least 
        implement a `split()` function. """
    splitter: Any = None
    fold: int = 0


@dataclass
class CrossValidator:
    name: str = MISSING
    splitter: Any = None
    fold: int = 0

    def _ensure_splitter(self):
        assert self.splitter is not None, "no splitter configured!"
        assert hasattr(
            self.splitter, "split"
        ), "cv splitter must have `split()` function."

    def split(self, X, y=None, groups=None) -> Generator[Tuple[List, List], None, None]:
        self._ensure_splitter()
        return self.splitter.split(X, y, groups)

    def get_split(self, X) -> Tuple[List, List]:
        self._ensure_splitter()
        splits = list(self.split(X))
        train_index, test_index = splits[self.fold]
        logger.info(
            f"{self.name}: using {len(train_index)} training samples and "
            + f"{len(test_index)} testing samples "
            + f"(fold={self.fold}, n_splits={len(splits)})"
        )
        return train_index, test_index

    def train_test_split(self, X, y) -> Tuple[List, List, List, List]:
        """Gets train/test split of current fold."""
        train_index, test_index = self.get_split(X)

        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        return X_train, X_test, y_train, y_test
