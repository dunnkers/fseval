from dataclasses import dataclass, field
from logging import Logger, getLogger
from typing import List

import pandas as pd
from codetiming import Timer
from fseval.pipeline.estimator import Estimator
from fseval.types import AbstractEstimator, Task
from humanfriendly import format_timespan


@dataclass
class Experiment(AbstractEstimator):
    estimators: List[AbstractEstimator] = field(default_factory=lambda: [])
    logger: Logger = getLogger(__name__)
    _enable_experiment_logging: bool = True

    def __post_init__(self):
        self.estimators = list(self._get_estimator())

    def _get_estimator(self):
        return []

    def _get_estimator_repr(self, estimator):
        return Estimator._get_estimator_repr(estimator)

    def _get_overrides_text(self, estimator):
        return ""

    def _logger(self, estimator):
        if self._enable_experiment_logging:
            return lambda text: getLogger(type(estimator).__name__).info(text)
        else:
            return lambda text: None

    def _step_text(self, step_name, step_number, estimator):
        step = step_number + 1
        n_steps = len(self.estimators)
        overrides_text = self._get_overrides_text(estimator)
        estimator_repr = self._get_estimator_repr(estimator)

        magenta = lambda text: f"\u001b[35m{text}\u001b[0m"
        cyan = lambda text: f"\u001b[36m{text}\u001b[0m"
        green = lambda text: f"\u001b[32m{text}\u001b[0m"
        yellow = lambda text: f"\u001b[33m{text}\u001b[0m"
        return lambda secs: (
            yellow(f"{overrides_text}")
            + f"{estimator_repr} ... {step_name} "
            + "in "
            + cyan(f"{format_timespan(secs)} ")
            + green("✓ ")
            + "("
            + magenta(f"step {step}/{n_steps}")
            + ")"
        )

    def _prepare_data(self, X, y):
        return X, y

    def fit(self, X, y) -> AbstractEstimator:
        X, y = self._prepare_data(X, y)
        self.fit_timer = Timer(name="fit")

        for step_number, estimator in enumerate(self.estimators):
            self.fit_timer.text = self._step_text("fit", step_number, estimator)
            self.fit_timer.logger = self._logger(estimator)
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
        X, y = self._prepare_data(X, y)
        scores = pd.DataFrame()

        for estimator in self.estimators:
            score = estimator.score(X, y)
            score = self._score_to_dataframe(score)
            scores = scores.append(score)

        return scores
