from typing import Dict, cast

from fseval.pipeline.estimator import Estimator
from fseval.types import AbstractEstimator, AbstractMetric, Callback


class UploadRankingScores(AbstractMetric):
    def score_bootstrap(
        self,
        ranker: AbstractEstimator,
        validator: AbstractEstimator,
        callbacks: Callback,
        scores: Dict,
        **kwargs,
    ) -> Dict:
        ranking_scores = scores["ranking"]

        ## upload ranking scores
        callbacks.on_table(ranking_scores, "ranking_scores")

        return scores
