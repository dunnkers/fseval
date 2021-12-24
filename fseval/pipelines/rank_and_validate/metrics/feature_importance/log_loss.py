from typing import Dict, Optional, Union, cast

import numpy as np
import pandas as pd
from fseval.pipeline.estimator import Estimator
from fseval.types import AbstractEstimator, AbstractMetric, Callback
from sklearn.metrics import log_loss

from ._normalize import normalize_feature_importances


class ImportanceLogLoss(AbstractMetric):
    """Computes the Log Loss metric between the estimated feature importances and
    the ground-truth feature importances, if they are available."""

    def score_ranking(
        self,
        scores: Union[Dict, pd.DataFrame],
        ranker: AbstractEstimator,
        bootstrap_state: int,
        callbacks: Callback,
        feature_importances: Optional[np.ndarray] = None,
    ) -> Union[Dict, pd.DataFrame]:
        ranker = cast(Estimator, ranker)

        scores["importance/log_loss"] = None

        if feature_importances is not None:
            y_true = np.asarray(feature_importances) > 0
            y_true = y_true.astype(int)
            y_pred = normalize_feature_importances(ranker.feature_importances_)
            score = log_loss(y_true, y_pred, labels=[0, 1])
            scores["importance/log_loss"] = score

        scores["importance/log_loss"] = scores["importance/log_loss"].astype(float)

        return scores
