from typing import Dict, Optional, Union, cast

import numpy as np
import pandas as pd
from fseval.pipeline.estimator import Estimator
from fseval.types import AbstractEstimator, AbstractMetric
from sklearn.metrics import log_loss

from ._normalize import normalize_feature_importances


class ImportanceLogLoss(AbstractMetric):
    """Computes the Log Loss metric between the estimated feature importances and
    the ground-truth feature importances, if they are available."""

    def score_ranking(
        self,
        ranker: AbstractEstimator,
        feature_importances: Optional[np.ndarray] = None,
    ) -> Optional[np.generic]:
        ranker = cast(Estimator, ranker)

        if feature_importances is not None:
            y_true = np.asarray(feature_importances) > 0
            y_true = y_true.astype(int)
            y_pred = normalize_feature_importances(ranker.feature_importances_)
            score = log_loss(y_true, y_pred, labels=[0, 1])

            return score
        else:
            return None
