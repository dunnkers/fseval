from dataclasses import dataclass
from fseval.config import RankerConfig
from typing import List
from sklearn.base import BaseEstimator
from hydra.utils import instantiate


@dataclass
class Ranker(RankerConfig, BaseEstimator):
    def __post_init__(self):
        task_estimator_mapping = dict(
            classification=self.classifier, regression=self.regressor
        )
        estimator_config = task_estimator_mapping[self.task.name]
        assert (
            estimator_config is not None
        ), f"{self.name} does not support {self.task.name} datasets!"
        self.estimator = instantiate(estimator_config)

    # def _get_estimator(self):
    #     task_estimator_mapping = dict(
    #         classification=self.classifier, regression=self.regressor
    #     )
    #     estimator = task_estimator_mapping[self.task.name]
    #     assert (
    #         estimator is not None
    #     ), f"{self.name} does not support {self.task.name} datasets!"
    #     return estimator

    def fit(self, X: List[List[float]], y: List) -> None:
        # estimator = self._get_estimator()
        self.estimator.fit(X, y)

    @property
    def feature_importances_(self) -> List[float]:
        # estimator = self._get_estimator()
        # return estimator.feature_importances_
        return self.estimator.feature_importances_
