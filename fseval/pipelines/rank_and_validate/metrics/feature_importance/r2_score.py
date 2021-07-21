from typing import Dict, Optional, Union, cast

import numpy as np
import pandas as pd
from fseval.pipeline.estimator import Estimator
from fseval.types import AbstractEstimator, AbstractMetric
from sklearn.metrics import r2_score

from ._normalize import normalize_feature_importances


class ImportanceR2Score(AbstractMetric):
    """Computes the R^2 (R-squared) metric between the estimated feature importances and
    the ground-truth feature importances, if they are available. Usually, the ground
    truth feature importances can be obtained by generating synthetic datasets, of which
    the ground-truth feature importances are known."""

    def score_ranking(
        self,
        ranker: AbstractEstimator,
        feature_importances: Optional[np.ndarray] = None,
    ) -> Optional[np.generic]:
        ranker = cast(Estimator, ranker)

        if feature_importances is not None:
            y_true = normalize_feature_importances(feature_importances)
            y_pred = normalize_feature_importances(ranker.feature_importances_)
            score = r2_score(y_true, y_pred)

            return score
        else:
            return None