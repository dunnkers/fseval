from dataclasses import dataclass, field
from logging import getLogger
from typing import List

import pandas as pd
from codetiming import Timer
from humanfriendly import format_timespan

from fseval.types import AbstractEstimator, Task

logger = getLogger(__name__)


@dataclass
class Experiment(AbstractEstimator):
    estimators: List[AbstractEstimator] = field(default_factory=lambda: [])

    def set_estimators(self, estimators: List[AbstractEstimator] = []):
        self.estimators = estimators

    @property
    def _scoring_metadata(self) -> List:
        return []

    def _get_scoring_metadata(self, estimator):
        metadata = {}

        params = estimator.get_params()
        for meta_attribute in self._scoring_metadata:
            metadata[meta_attribute] = params.get(meta_attribute, None)

        if "fit_time" in self._scoring_metadata:
            metadata["fit_time"] = getattr(estimator, "_fit_time_elapsed", None)

        return metadata

    def _timer_text(self, step_name, step_number, estimator):
        n_steps = len(self.estimators)

        step_text = ""
        for meta_key, meta_value in self._get_scoring_metadata(estimator).items():
            if meta_value is not None:
                step_text += f"{meta_key}={meta_value} "
        step_text += "\t"

        step_text += f"{step_name} step {step_number + 1}/{n_steps} took"

        return lambda secs: f"{step_text} {format_timespan(secs)}"

    def fit(self, X, y) -> AbstractEstimator:
        self.fit_timer = Timer(name="fit", logger=logger.info)

        for step_number, estimator in enumerate(self.estimators):
            self.fit_timer.text = self._timer_text("fit", step_number, estimator)
            self.fit_timer.start()
            estimator.fit(X, y)
            self.fit_timer.stop()
            setattr(estimator, "_fit_time_elapsed", self.fit_timer.last)

        return self

    def transform(self, X, y):
        ...

    def fit_transform(self, X, y):
        ...

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

        return scores
