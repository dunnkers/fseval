import numpy as np
import pandas as pd


class ImportanceMetricMixin:
    @property
    def estimated_feature_importances(self):
        """Normalized estimated feature importances. The summation of the importances
        vector is always 1."""

        # get ranker feature importances, check whether all components > 0
        feature_importances = np.asarray(self.ranker.feature_importances_)
        assert not (
            feature_importances < 0
        ).any(), "estimated feature importances must be strictly positive."

        # normalize
        feature_importances = feature_importances / sum(feature_importances)

        return feature_importances

    @property
    def ground_truth_feature_importances(self):
        """Ground truth normalized feature importances."""
        X_importances = np.asarray(self.X_importances)
        X_importances = X_importances / sum(X_importances)

        return X_importances

    ...
