from dataclasses import dataclass
from fseval.config import RankerConfig
from typing import List
from sklearn.base import BaseEstimator


@dataclass
class Ranker(RankerConfig, BaseEstimator):
    def fit(self, X: List[List[float]], y: List) -> None:
        raise NotImplementedError

    @property
    def feature_importances_(self) -> List[float]:
        raise NotImplementedError
