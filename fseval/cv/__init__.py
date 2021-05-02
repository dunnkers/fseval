import logging
from typing import Tuple, List, Generator
from dataclasses import dataclass
from fseval.config import CrossValidatorConfig
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


@dataclass
class CrossValidator(CrossValidatorConfig, BaseEstimator):
    def split(self, X, y=None, groups=None) -> Generator[Tuple[List, List], None, None]:
        raise NotImplementedError

    def get_split(self, X) -> Tuple[List, List]:
        splits = list(self.split(X))
        train_index, test_index = splits[self.fold]
        logger.info(
            f"using {len(train_index)} training samples and "
            + f"{len(test_index)} testing samples "
            + f"(fold={self.fold}, n_splits={self.n_splits})"
        )
        return train_index, test_index
