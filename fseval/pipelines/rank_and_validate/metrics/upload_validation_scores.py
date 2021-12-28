from typing import Dict

from fseval.types import AbstractEstimator, AbstractMetric, Callback


class UploadValidationScores(AbstractMetric):
    def score_bootstrap(
        self,
        ranker: AbstractEstimator,
        validator: AbstractEstimator,
        callbacks: Callback,
        scores: Dict,
        **kwargs,
    ) -> Dict:
        validation_scores = scores["validation"]

        ## upload validation scores
        callbacks.on_table(validation_scores, "validation_scores")

        return scores
