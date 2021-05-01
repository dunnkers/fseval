from dataclasses import dataclass
from fseval.config import RankerConfig
from typing import List

@dataclass
class Ranker(RankerConfig):
    def fit(self, X: List[List[float]], y: List) -> None:
        raise NotImplementedError

    @property
    def feature_importances_(self) -> List[float]: raise NotImplementedError