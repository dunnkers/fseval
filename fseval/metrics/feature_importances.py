from typing import Dict, Optional, Union, cast

import numpy as np
import pandas as pd

from fseval.pipeline.estimator import Estimator
from fseval.types import AbstractEstimator, AbstractMetric, Callback


class UploadFeatureImportances(AbstractMetric):
    def _build_table(self, feature_vector: np.ndarray):
        """Takes a feature importances vector of type (n_features) or
        (n_classes, n_features)."""

        # feature_vector is of form (n_features)
        if feature_vector.ndim == 1:
            indices = np.arange(1, len(feature_vector) + 1)
            df = pd.DataFrame(
                {
                    "feature_importances": feature_vector,
                    "feature_index": indices,
                    "class": None,
                }
            )
            return df

        # feature_vector is of form (n_classes, n_features)
        elif feature_vector.ndim == 2:
            df = pd.DataFrame()

            for class_index, feature_vector_class in enumerate(feature_vector):
                indices = np.arange(1, len(feature_vector_class) + 1)
                df_class = pd.DataFrame(
                    {
                        "feature_importances": feature_vector_class,
                        "feature_index": indices,
                        "class": class_index,
                    }
                )
                df = pd.concat([df, df_class])

            return df

        else:
            raise ValueError(
                "`feature_importances` must be either 1- or 2-dimensional."
            )

    def _normalize_feature_importances(self, feature_importances: np.ndarray):
        """Normalized feature importances. The summation of the importances
        vector is always 1."""

        feature_importances = np.asarray(feature_importances)

        # get ranker feature importances, check whether all components > 0
        assert not (feature_importances < 0).any(), (
            "Estimated or ground-truth feature importances must be strictly positive."
            + " Some feature importance scores were negative."
        )

        # normalize
        if feature_importances.ndim == 1:
            feature_importances = feature_importances / sum(feature_importances)
        elif feature_importances.ndim == 2:
            feature_importances_rowsum = feature_importances.sum(axis=1, keepdims=True)
            feature_importances = feature_importances / feature_importances_rowsum
        else:
            raise ValueError(
                "`feature_importances` must be either 1- or 2-dimensional."
            )

        return feature_importances

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
            estimated = self._normalize_feature_importances(ranker.feature_importances_)
            estimated_df = self._build_table(estimated)
            estimated_df["group"] = "estimated"
            estimated_df["bootstrap_state"] = bootstrap_state
            table = pd.concat([table, estimated_df])

        if feature_importances is not None:
            ground_truth = self._normalize_feature_importances(feature_importances)
            ground_truth_df = self._build_table(ground_truth)
            ground_truth_df["group"] = "ground_truth"
            ground_truth_df["bootstrap_state"] = bootstrap_state
            table = pd.concat([table, ground_truth_df])

        if not table.empty:
            callbacks.on_table(table, "feature_importances")

        return scores
