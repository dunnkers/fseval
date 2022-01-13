from dataclasses import dataclass
from typing import Dict, Union, cast

import numpy as np
import pandas as pd
from omegaconf import MISSING
from sklearn.feature_selection import SelectFromModel

from fseval.pipeline.estimator import Estimator
from fseval.types import IncompatibilityError

from .._experiment import Experiment
from ._config import RankAndValidatePipeline


@dataclass
class SubsetValidator(Experiment, RankAndValidatePipeline):
    """Validates one feature subset using a given validation estimator. i.e. it first
    performs feature selection using the ranking made available in the fitted ranker,
    `self.ranker`, and then fits/scores an estimator on that subset."""

    bootstrap_state: int = MISSING
    n_features_to_select: int = MISSING

    def __post_init__(self):
        if not self.validator.estimates_target:
            raise IncompatibilityError(
                f"{self.validator.name} does not predict targets: "
                + "this estimator cannot be used as a validator."
            )

        super(SubsetValidator, self).__post_init__()

    def _get_estimator(self):
        yield self.validator

    def _logger(self, estimator):
        return lambda text: None

    def _get_feature_importances(self, estimator: Estimator):
        if estimator.estimates_feature_importances:
            return estimator.feature_importances_
        elif estimator.estimates_feature_ranking:
            return estimator.feature_ranking_
        else:
            raise ValueError(
                f"could not resolve feature_importances vector on {estimator.name}."
            )

    def _prepare_data(self, X, y):
        # select n features: perform feature selection
        selector = SelectFromModel(
            estimator=self.ranker,
            threshold=-np.inf,
            max_features=self.n_features_to_select,
            importance_getter=self._get_feature_importances,
            prefit=True,
        )
        X = selector.transform(X)
        return X, y

    @property
    def _cache_filename(self):
        override = f"bootstrap_state={self.bootstrap_state}"
        override += f",n_features_to_select={self.n_features_to_select}"
        filename = f"validation[{override}].pickle"

        return filename

    def prefit(self):
        self.validator._load_cache(self._cache_filename, self.storage)

    def fit(self, X, y):
        super(SubsetValidator, self).fit(X, y)

    def postfit(self):
        self.validator._save_cache(self._cache_filename, self.storage)

    def score(self, X, y, **kwargs) -> Union[Dict, pd.DataFrame, np.generic, None]:
        """Compute validator score. Uses the `score()` function configured in the
        validator itself. For example, k-NN has a `score()` function that uses the
        `accuracy` score. To customize, override the `score()` function in the
        validation estimator."""

        # Compute validator score. Uses estimator's `score()` function.
        validator_score = super(SubsetValidator, self).score(X, y)
        assert np.isscalar(validator_score), (
            f"'{self.validator.name}' validator score must be a scalar. That is, "
            + "it must be an int, float, string or boolean. The validator score is "
            + f"whatever is returned by `{self.validator.score}`."
        )
        validator_score = cast(np.generic, validator_score)

        # Attach score scalar to scoring object.
        scores_dict = {}
        scores_dict["n_features_to_select"] = self.n_features_to_select
        scores_dict["fit_time"] = self.validator.fit_time_
        scores_dict["score"] = validator_score  # type: ignore

        # Convert to DataFrame
        scores = pd.DataFrame([scores_dict])

        # Add custom metrics
        for metric_name, metric_class in self.metrics.items():
            scores_metric = metric_class.score_subset(scores, self.validator, X, y, self.callbacks)  # type: ignore

            if scores_metric is not None:
                scores = scores_metric

        return scores
