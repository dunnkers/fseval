from typing import Dict, Optional, Union, cast

import numpy as np
import pandas as pd
from fseval.pipeline.estimator import Estimator
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
        ranker = cast(Estimator, ranker)
        validator = cast(Estimator, validator)

        validation_scores = scores["validation"]

        ## upload validation scores
        callbacks.on_table(validation_scores, "validation_scores")

        return scores
