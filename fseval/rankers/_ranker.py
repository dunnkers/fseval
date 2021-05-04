from dataclasses import dataclass, field
from fseval.config import RankerConfig
from typing import List, Any, Optional
import numpy as np
from sklearn.base import ClassifierMixin
from fseval.base import Configurable
from sklearn.metrics import log_loss


@dataclass
class Ranker(RankerConfig, ClassifierMixin, Configurable):
    """
    Feature ranker. Given a dataset X and its target variables, a feature ranker
    constructs a feature importance score for each feature. The ranker is considered
    a binary classifier, allowing us to use all sklearn utilities accordingly.
    """

    @classmethod
    def _get_config_names(cls):
        params = super()._get_param_names()
        params.remove("classifier")
        params.remove("regressor")
        params.append("estimator")
        return params

    @property
    def estimator(self) -> Any:
        # choose estimator according to task (classifier or regressor)
        estimators = dict(classification=self.classifier, regression=self.regressor)
        estimator = estimators[self.task.name]

        # make sure ranker has the correct estimator defined
        assert (
            estimator is not None
        ), f"{self.name} does not support {self.task.name} datasets!"

        return estimator

    def fit(self, X: List[List[float]], y: List) -> None:
        self.estimator.fit(X, y)

    def predict(self, X=None) -> np.ndarray:
        """
        Returns:
            y: array-like of shape (n_features,)
        """
        assert hasattr(
            self.estimator, "selected_features_"
        ), f"{self.name} ranker does not select subsets; but `predict()` was still called."
        return self.estimator.selected_features_

    def predict_proba(self, X: List[List[float]] = None) -> List:
        """
        Feature importance scores. Global or per-instance.

        Returns:
            P: ndarray of shape (n_features,). Importance scores of features.
        """
        if X is not None and not hasattr(self, "instance_based"):
            raise NotImplementedError(
                "instance-based feature ranking not supported yet"
            )
        return self.feature_importances_

    def score(self, X: List, y: List):
        importance_scores = self.predict_proba(X)

        assert y != None, "must provide true labels `y` in order to score Ranker."
        assert not np.ndim(y) > np.ndim(
            importance_scores
        ), f"""cannot use an instance-based relevant features ground truth with 
            {self.name} ranker: the ranker did not return an instance-based 
            feature ranking."""

        return log_loss(y, importance_scores, labels=[0, 1])

    @property
    def feature_importances_(self) -> List:
        return self.estimator.feature_importances_
