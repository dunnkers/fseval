from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, cast

import pandas as pd
from fseval.base import AbstractEstimator, Task


@dataclass
class AbstractExperiment(AbstractEstimator):
    estimators: List[AbstractEstimator] = field(default_factory=lambda: [])

    def set_estimators(self, estimators: List[AbstractEstimator] = []):
        self.estimators = estimators

    @property
    def _scoring_metadata(self) -> List:
        return []

    def _get_scoring_metadata(self, estimator):
        metadata = {}

        for meta_attribute in self._scoring_metadata:
            params = estimator.get_params()
            metadata[meta_attribute] = params.get(meta_attribute, None)

        return metadata

    def fit(self, X, y) -> AbstractEstimator:
        for estimator in self.estimators:
            estimator.fit(X, y)

        return self

    def transform(self, X, y):
        ...

    def fit_transform(self, X, y):
        for estimator in self.estimators:
            estimator.fit_transform(X, y)

    def _score_to_dataframe(self, score):
        if isinstance(score, pd.DataFrame):
            return score
        elif isinstance(score, float) or isinstance(score, int):
            return pd.DataFrame([{"score": score}])
        else:
            raise ValueError(f"illegal score type received: {type(score)}")

    def score(self, X, y) -> pd.DataFrame:
        scores = pd.DataFrame()

        for estimator in self.estimators:
            score = estimator.score(X, y)
            score_df = self._score_to_dataframe(score)

            metadata = self._get_scoring_metadata(estimator)
            score_df = score_df.assign(**metadata)

            scores = scores.append(score_df)

        return pd.DataFrame(scores)
