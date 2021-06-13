from dataclasses import dataclass

import numpy as np
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
        self.validator._load_cache(self._cache_filename, self.storage_provider)

    def fit(self, X, y):
        super(SubsetValidator, self).fit(X, y)

    def postfit(self):
        self.validator._save_cache(self._cache_filename, self.storage_provider)

    def score(self, X, y, **kwargs):
        score = super(SubsetValidator, self).score(X, y)
        score["n_features_to_select"] = self.n_features_to_select
        score["fit_time"] = self.validator.fit_time_
        return score
