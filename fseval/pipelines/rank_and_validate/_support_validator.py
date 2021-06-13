from dataclasses import dataclass

import numpy as np
from fseval.pipeline.estimator import Estimator
from fseval.types import IncompatibilityError
from omegaconf import MISSING
from sklearn.feature_selection import SelectFromModel

from .._experiment import Experiment
from ._config import RankAndValidatePipeline
from ._subset_validator import SubsetValidator


@dataclass
class SupportValidator(SubsetValidator):
    """Validates a feature support vector, i.e. a feature subset."""

    bootstrap_state: int = MISSING
    n_features_to_select: int = -1  # disable

    def _prepare_data(self, X, y):
        feature_support = getattr(self.ranker, "feature_support_", None)
        assert feature_support is not None, "ranker must have support attribute"
        assert isinstance(
            feature_support, np.ndarray
        ), "feature support array must be a numpy ndarray"

        # make sure support vector is boolean-valued
        feature_support = feature_support.astype(bool)
        self.subset_size = np.sum(feature_support)

        # select feature subset
        X = X[:, feature_support]

        return X, y

    @property
    def _cache_filename(self):
        override = f"bootstrap_state={self.bootstrap_state}"
        filename = f"support[{override}].pickle"

        return filename

    def score(self, X, y, **kwargs):
        score = super(SubsetValidator, self).score(X, y)
        score["subset_size"] = self.subset_size
        score["fit_time"] = self.validator.fit_time_
        return score
