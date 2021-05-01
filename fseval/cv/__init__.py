from typing import Tuple, List, Generator
from dataclasses import dataclass
from fseval.config import CrossValidatorConfig


@dataclass
class CrossValidator(CrossValidatorConfig):
    def split(self, X, y=None, groups=None) -> Generator[Tuple[List, List], None, None]:
        raise NotImplementedError

    def get_split(self, X) -> Tuple[List, List]:
        splits = list(self.split(X))
        train_index, test_index = splits[self.fold]
        return train_index, test_index
