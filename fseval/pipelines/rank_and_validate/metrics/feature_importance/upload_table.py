from typing import Dict, Optional, Union, cast

import numpy as np
import pandas as pd
from fseval.pipeline.estimator import Estimator
from fseval.types import AbstractEstimator, AbstractMetric, Callback

from ._normalize import normalize_feature_importances


class UploadFeatureImportances(AbstractMetric):
    def _build_table(self, feature_vector: np.ndarray):
        indices = np.arange(1, len(feature_vector) + 1)
        df = pd.DataFrame(
            {"feature_importances": feature_vector, "feature_index": indices}
        )

        return df

    def score_ranking(
        self,
        scores: Union[Dict, pd.DataFrame],
        ranker: AbstractEstimator,
        bootstrap_state: int,
        callbacks: Callback,
        feature_importances: Optional[np.ndarray] = None,
    ) -> Union[Dict, pd.DataFrame]:
        ranker = cast(Estimator, ranker)

        table = pd.DataFrame()

        if ranker.estimates_feature_importances:
            estimated = normalize_feature_importances(ranker.feature_importances_)
            estimated_df = self._build_table(estimated)
            estimated_df["group"] = "estimated"
            estimated_df["bootstrap_state"] = bootstrap_state
            table = table.append(estimated_df)

        if feature_importances is not None:
            ground_truth = normalize_feature_importances(feature_importances)
            ground_truth_df = self._build_table(ground_truth)
            ground_truth_df["group"] = "ground_truth"
            ground_truth_df["bootstrap_state"] = bootstrap_state
            table = table.append(ground_truth_df)

        if not table.empty:
            callbacks.on_table(table, "feature_importances")

        return scores
