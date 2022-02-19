from dataclasses import dataclass
from typing import Dict, Union, cast

import numpy as np
import pandas as pd
from omegaconf import MISSING

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
        """This function overrides `_cache_filename` in `SubsetValidator`."""

        override = f"bootstrap_state={self.bootstrap_state}"
        filename = f"support[{override}].pickle"

        return filename

    def score(self, X, y, **kwargs) -> Union[Dict, pd.DataFrame, np.generic, None]:
        # See `SubsetValidator.score()`. This uses the validation estimator's `score()`
        # function.
        # fmt: off
        validator_score = super(SubsetValidator, self).score(X, y)  # lgtm [py/super-not-enclosing-class]
        # fmt: on
        assert np.isscalar(validator_score), (
            f"'{self.validator.name}' validator score must be a scalar. That is, "
            + "it must be an int, float, string or boolean. The validator score is "
            + f"whatever is returned by `{self.validator.score}`."
        )
        validator_score = cast(np.generic, validator_score)

        # Put scores in object
        scores_dict = {}
        scores_dict["score"] = validator_score  # type: ignore
        scores_dict["subset_size"] = self.subset_size
        scores_dict["fit_time"] = self.validator.fit_time_

        # convert to DataFrame
        scores = pd.DataFrame([scores_dict])

        # add custom metrics
        X_, y_ = self._prepare_data(X, y)

        for metric_name, metric_class in self.metrics.items():
            scores_metric = metric_class.score_support(  # type: ignore
                scores, self.validator, X_, y_, self.callbacks
            )  # type: ignore

            if scores_metric is not None:
                scores = scores_metric

        return scores
