from typing import Dict, Optional, Union, cast

import numpy as np
import pandas as pd
from fseval.pipeline.estimator import Estimator
from fseval.types import AbstractEstimator, AbstractMetric
from sklearn.metrics import r2_score


class ImportanceR2Score(AbstractMetric):
    """Computes the R^2 (R-squared) metric between the estimated feature importances and
    the ground-truth feature importances, if they are available. Usually, the ground
    truth feature importances can be obtained by generating synthetic datasets, of which
    the ground-truth feature importances are known."""

    def normalized_feature_importances(self, feature_importances):
        """Normalized feature importances. The summation of the importances
        vector is always 1."""

        # get ranker feature importances, check whether all components > 0
        feature_importances = np.asarray(feature_importances)
        assert not (
            feature_importances < 0
        ).any(), (
            "estimated or ground-truth feature importances must be strictly positive."
        )

        # normalize
        feature_importances = feature_importances / sum(feature_importances)

        return feature_importances

    def score_ranking(
        self,
        ranker: AbstractEstimator,
        feature_importances: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Union[Dict, pd.DataFrame, int, float, None]:
        ranker = cast(Estimator, ranker)
        score = {}

        if feature_importances is not None:
            y_true = self.normalized_feature_importances(feature_importances)
            y_pred = self.normalized_feature_importances(ranker.feature_importances_)
            score["importance/r2_score"] = r2_score(y_true, y_pred)
        else:
            score["importance/r2_score"] = None

        return score
