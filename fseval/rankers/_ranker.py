from dataclasses import dataclass, field
from fseval.config import RankerConfig
from typing import List, Any, Optional
import numpy as np
from sklearn.base import ClassifierMixin
from fseval.base import ConfigurableEstimator
from sklearn.metrics import log_loss


@dataclass
class Ranker(RankerConfig, ConfigurableEstimator, ClassifierMixin):
    """
    Feature ranker. Given a dataset X and its target variables, a feature ranker
    constructs a feature importance score for each feature. The ranker is considered
    a binary classifier, allowing us to use all sklearn utilities accordingly.
    """

    def predict(self, X: List[List[float]] = None) -> List:
        """
        Returns:
            y: array-like of shape (n_features,)
        """
        # assert hasattr(
        #     self.estimator, "selected_features_"
        # ), f"{self.name} ranker does not select subsets; but `predict()` was still called."
        if not hasattr(self.estimator, "selected_features_"):
            # if no subset selection method available, just remove any feature that
            # ranks below the average score.
            scores = np.asarray(self.estimator.feature_importances_)
            subset = scores > sum(scores) / len(scores)
            subset = subset.astype(int)
            return subset

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

    def score(self, X: List[List[float]], y: List, sample_weight=None) -> float:
        importance_scores = self.predict_proba(X)

        assert y != None, "must provide true labels `y` in order to score Ranker."
        assert not np.ndim(y) > np.ndim(
            importance_scores
        ), f"""cannot use an instance-based relevant features ground truth with 
            {self.name} ranker: the ranker did not return an instance-based 
            feature ranking."""

        return log_loss(y, importance_scores, labels=[0, 1])
