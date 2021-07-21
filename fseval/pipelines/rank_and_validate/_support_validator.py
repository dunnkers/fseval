from dataclasses import dataclass
from typing import Dict, Union, cast

import numpy as np
import pandas as pd
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

    def score(self, X, y, **kwargs) -> Union[Dict, pd.DataFrame, np.generic, None]:
        # See `SubsetValidator.score()`. This uses the validation estimator's `score()`
        # function.
        validator_score = super(SubsetValidator, self).score(X, y)
        assert np.isscalar(validator_score), (
            f"'{self.validator.name}' validator score must be a scalar. That is, "
            + "it must be an int, float, string or boolean. The validator score is "
            + f"whatever is returned by `{self.validator.score}`."
        )
        validator_score = cast(np.generic, validator_score)

        # Put scores in object
        scores = {}
        scores["score"] = validator_score  # type: ignore
        scores["subset_size"] = self.subset_size
        scores["fit_time"] = self.validator.fit_time_

        # add custom metrics
        for metric_name, metric_class in self.metrics.items():
            X, y = self._prepare_data(X, y)
            scores[metric_name] = metric_class.score_support(
                self.validator, X, y
            )  # type: ignore

        # convert to DataFrame
        scores_df = pd.DataFrame([scores])

        return scores_df
