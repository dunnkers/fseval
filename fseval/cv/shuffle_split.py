from typing import Tuple, List, Generator
from sklearn.model_selection import ShuffleSplit as sklearn_ShuffleSplit
from dataclasses import dataclass
from fseval.cv import CrossValidator


@dataclass
class ShuffleSplit(CrossValidator):
    def __post_init__(self):
        self.estimator = sklearn_ShuffleSplit(
            n_splits=self.n_splits,
            test_size=self.test_size,
            train_size=self.train_size,
            random_state=self.random_state,
        )

    def split(self, X, y=None, groups=None) -> Generator[Tuple[List, List], None, None]:
        return self.estimator.split(X, y, groups)
