from dataclasses import dataclass
from fseval.config import RankerConfig
from typing import List, Any
from sklearn.base import BaseEstimator


@dataclass
class Ranker(RankerConfig, BaseEstimator):
    estimator: Any = None

    def __post_init__(self):
        # choose estimator according to task (classifier or regressor)
        task_estimator_mapping = dict(
            classification=self.classifier, regression=self.regressor
        )
        estimator = task_estimator_mapping[self.task.name]

        # make sure ranker has the correct estimator defined
        assert (
            estimator is not None
        ), f"{self.name} does not support {self.task.name} datasets!"

        """ remove these: `BaseEstimator.get_params()` tries to recursively get 
            params from these objects. i.e., hasattr(self.classifier, 'get_params') 
            returns true. """
        self.classifier = None
        self.regressor = None

        # set `estimator` attribute for easy access.
        self.estimator = estimator

    def fit(self, X: List[List[float]], y: List) -> None:
        self.estimator.fit(X, y)

    @property
    def feature_importances_(self) -> List[float]:
        return self.estimator.feature_importances_
