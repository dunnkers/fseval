from dataclasses import dataclass
from logging import Logger, getLogger
from typing import Optional

import numpy as np
import pandas as pd
from fseval.types import IncompatibilityError, TerminalColor
from omegaconf import MISSING
from sklearn.metrics import accuracy_score, log_loss, r2_score

from .._experiment import Experiment
from ._config import RankAndValidatePipeline


@dataclass
class RankingValidator(Experiment, RankAndValidatePipeline):
    bootstrap_state: int = MISSING

    logger: Logger = getLogger(__name__)

    def __post_init__(self):
        if not (
            self.ranker.estimates_feature_importances
            or self.ranker.estimates_feature_ranking
            or self.ranker.estimates_feature_support
        ):
            raise IncompatibilityError(
                f"{self.ranker.name} performs no form of feature ranking: "
                + "this estimator cannot be used as a ranker."
            )

        super(RankingValidator, self).__post_init__()

    @property
    def _cache_filename(self):
        override = f"bootstrap_state={self.bootstrap_state}"
        filename = f"ranking[{override}].pickle"

        return filename

    def _get_estimator(self):
        yield self.ranker

    def prefit(self):
        self.ranker._load_cache(self._cache_filename, self.storage_provider)

    def fit(self, X, y):
        self.logger.info(f"fitting ranker: " + TerminalColor.yellow(self.ranker.name))

        super(RankingValidator, self).fit(X, y)

    def postfit(self):
        self.ranker._save_cache(self._cache_filename, self.storage_provider)

    def _scores_to_ranking(self, scores):
        """Converts a scoring vector to a ranking vector, or standardizes an existing
        feature ranking vector. e.g.:

        ```
        [0.8, 0.1, 0.9, 0.0]
        ```
        is converted to
        ```
        [3, 2, 4, 1]
        ```
        """

        _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
        ranking_inverse = np.zeros_like(counts)
        ranking_inverse[1:] = counts[:-1].cumsum()
        ranking = ranking_inverse[inverse] + 1

        return ranking

    @property
    def estimated_feature_importances(self):
        """Normalized estimated feature importances. The summation of the importances
        vector is always 1."""

        feature_importances = np.asarray(self.ranker.feature_importances_)
        feature_importances = feature_importances / sum(feature_importances)

        return feature_importances

    @property
    def ground_truth_feature_importances(self):
        """Ground truth normalized feature importances."""
        X_importances = np.asarray(self.X_importances)
        X_importances = X_importances / sum(X_importances)

        return X_importances

    @property
    def estimated_feature_support(self):
        """Normalized estimated feature support. Returns the feature support of each
        feature as an integer. 0 means feature was not selected, 1 means the feature was
        selected."""

        feature_support = np.asarray(self.ranker.feature_support_, dtype=bool)
        feature_support = feature_support.astype(int)

        return feature_support

    @property
    def ground_truth_feature_support(self):
        """Ground truth feature support. Simply checks whether the ground truth feature
        support is more than one, i.e., feature_importance > 0. Returns a 0 for 'should
        not select this feature' and 1 for 'should select this feature'."""
        X_importances = np.asarray(self.X_importances)
        X_importances = X_importances > 0
        X_importances = X_importances.astype(int)

        return X_importances

    @property
    def estimated_feature_ranking(self):
        """Normalized estimated feature ranking. GrabS ranking through either (1)
        `ranking_` or (2) `feature_importances_`"""

        assert (
            self.ranker.estimates_feature_ranking
            or self.ranker.estimates_feature_importances
        ), "ranker must estimate at least one of importance or ranking."

        ranking = None
        if self.ranker.estimates_feature_ranking:
            ranking = self.ranker.feature_ranking_
        elif self.ranker.estimates_feature_importances:
            ranking = self.ranker.feature_importances_

        # predicted feature ranking, re-ordered and normalized.
        feature_ranking = self._scores_to_ranking(ranking)
        feature_ranking = feature_ranking / sum(feature_ranking)

        return feature_ranking

    @property
    def ground_truth_feature_ranking(self):
        """Ground truth normalized feature ranking. Normalizes the ranking using the
        `_scores_to_ranking` function, which converts a vector of any arbitrary scaling
        to a proportionally scaled probability vector."""

        X_importances = np.asarray(self.X_importances)
        X_importances = self._scores_to_ranking(X_importances)
        X_importances = X_importances / sum(X_importances)

        return X_importances

    def _score_with_feature_importances(self, score):
        """Scores this feature ranker with the available dataset ground-truth relevant
        features, which are to be known apriori. Supports three types of feature rankings:
        - a real-valued feature importance vector
        - a boolean-valued feature support vector
        - an integer-valued feature ranking vector."""

        ### Feature importances
        if self.ranker.estimates_feature_importances:
            # r2 score
            y_true = self.ground_truth_feature_importances
            y_pred = self.estimated_feature_importances
            score["importance/r2_score"] = r2_score(y_true, y_pred)

            # log loss
            y_true = self.ground_truth_feature_support
            score["importance/log_loss"] = log_loss(y_true, y_pred, labels=[0, 1])

        ### Feature support
        if self.ranker.estimates_feature_support:
            # accuracy
            y_true = self.ground_truth_feature_support
            y_pred = self.estimated_feature_support
            score["support/accuracy"] = accuracy_score(y_true, y_pred)

        ### Feature ranking
        if (
            self.ranker.estimates_feature_importances
            or self.ranker.estimates_feature_ranking
        ):
            # re-ordered and normalized rankings.
            y_true = self.ground_truth_feature_ranking
            y_pred = self.estimated_feature_ranking

            # in r2 score, only consider **relevant** features, not irrelevant ones. in
            # this way, when `X_importances = [0, 2, 4, 0, 0]` we do not get misleadingly
            # high scores.
            sample_weight = np.ones_like(self.X_importances)
            sample_weight[self.X_importances == 0] = 0.0

            # r2 score
            score["ranking/r2_score"] = r2_score(
                y_true, y_pred, sample_weight=sample_weight
            )

    def score(self, X, y, **kwargs):
        """Scores a feature ranker, if a ground-truth on the desired dataset
        feature importances is available. If this is the case, the estimated normalized
        feature importances are compared to the desired ones using two metrics:
        log loss and the R^2 score. Whilst the log loss converts the ground-truth
        desired feature rankings to a binary value, 0/1, the R^2 score always works."""

        score = {
            "fit_time": self.ranker.fit_time_,
            "bootstrap_state": self.bootstrap_state,
        }

        # score using ground truth, if available.
        X_importances: Optional[np.ndarray] = kwargs.get("feature_importances")
        self.X_importances = X_importances  # store for later use

        if X_importances is not None:
            assert np.ndim(X_importances) == 1, "instance-based not supported yet."
            self._score_with_feature_importances(score)

        # put a in a dataframe so can be easily merged with other pipeline scores
        scores = pd.DataFrame([score])
        return scores
