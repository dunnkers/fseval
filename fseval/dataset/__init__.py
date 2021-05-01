from typing import Tuple, List
from dataclasses import dataclass
from fseval.config import DatasetConfig


@dataclass
class Dataset(DatasetConfig):
    def load(self) -> Tuple[List, List]:
        raise NotImplementedError
