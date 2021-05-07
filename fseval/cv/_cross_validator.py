import logging
from dataclasses import dataclass
from typing import Generator, List, Tuple

from fseval.base import Configurable
from fseval.config import CrossValidatorConfig

logger = logging.getLogger(__name__)


@dataclass
class CrossValidator(CrossValidatorConfig, Configurable):
    def _ensure_splitter(self):
        assert self.splitter is not None, "no splitter configured!"

    def split(self, X, y=None, groups=None) -> Generator[Tuple[List, List], None, None]:
        self._ensure_splitter()
        return self.splitter.split(X, y, groups)

    def get_split(self, X) -> Tuple[List, List]:
        self._ensure_splitter()
        splits = list(self.split(X))
        train_index, test_index = splits[self.fold]
        logger.info(
            f"using {len(train_index)} training samples and "
            + f"{len(test_index)} testing samples "
            + f"(fold={self.fold}, n_splits={len(splits)})"
        )
        return train_index, test_index
