from dataclasses import dataclass
from logging import Logger, getLogger

import numpy as np
import pandas as pd
from fseval.types import IncompatibilityError
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

    def _score_with_feature_importances(self, score):
        """Scores this feature ranker with the available dataset ground-truth relevant
        features, which are to be known apriori. Supports three types of feature rankings:
        - a real-valued feature importance vector
        - a boolean-valued feature support vector
        - an integer-valued feature ranking vector."""

        X_importances = self.dataset.feature_importances

        ### Feature importances
        if self.ranker.estimates_feature_importances:
            # predicted feature importances, normalized.
            y_pred = np.asarray(self.ranker.feature_importances_)
            y_pred = y_pred / sum(y_pred)

            # r2 score
            y_true = X_importances
            score["importance.r2_score"] = r2_score(y_true, y_pred)

            # log loss
            y_true = X_importances > 0
            score["importance.log_loss"] = log_loss(y_true, y_pred, labels=[0, 1])

        ### Feature support
        if self.ranker.estimates_feature_support:
            # predicted feature support
            y_pred = np.asarray(self.ranker.feature_support_, dtype=bool)

            # accuracy
            y_true = X_importances > 0
            score["support.accuracy"] = accuracy_score(y_true, y_pred)

        ### Feature ranking
        # grab ranking through either (1) `ranking_` or (2) `feature_importances_`
        ranking = None
        if self.ranker.estimates_feature_ranking:
            ranking = self.ranker.feature_ranking_
        elif self.ranker.estimates_feature_importances:
            ranking = self.ranker.feature_importances_

        # compute ranking r2 score
        if ranking is not None:
            # predicted feature ranking, re-ordered and normalized.
            y_pred = self._scores_to_ranking(ranking)
            y_pred = y_pred / sum(y_pred)

            # convert ground-truth to a ranking as well.
            y_true = self._scores_to_ranking(X_importances)
            y_true = y_true / sum(y_true)

            # in r2 score, only consider **relevant** features, not irrelevant ones. in
            # this way, when `X_importances = [0, 2, 4, 0, 0]` we do not get misleadingly
            # high scores because the ranking also
            sample_weight = np.ones_like(X_importances)
            sample_weight[X_importances == 0] = 0.0

            # r2 score
            score["ranking.r2_score"] = r2_score(
                y_true, y_pred, sample_weight=sample_weight
            )

    def score(self, X, y):
        """Scores a feature ranker, if a ground-truth on the desired dataset
        feature importances is available. If this is the case, the estimated normalized
        feature importances are compared to the desired ones using two metrics:
        log loss and the R^2 score. Whilst the log loss converts the ground-truth
        desired feature rankings to a binary value, 0/1, the R^2 score always works."""

        score = {
            "fit_time": self.ranker.fit_time_,
            "bootstrap_state": self.bootstrap_state,
        }

        if self.dataset.feature_importances is not None:
            assert (
                np.ndim(self.dataset.feature_importances) == 1
            ), "instance-based not supported yet."

            self._score_with_feature_importances(score)

        # put a in a dataframe so can be easily merged with other pipeline scores
        scores = pd.DataFrame([score])
        return scores
