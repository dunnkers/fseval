from dataclasses import dataclass, field
from logging import Logger, getLogger
from time import perf_counter
from typing import List

import pandas as pd
from humanfriendly import format_timespan

from fseval.pipeline.estimator import Estimator
from fseval.types import AbstractEstimator, Task, TerminalColor


@dataclass
class Experiment(AbstractEstimator):
    estimators: List[AbstractEstimator] = field(default_factory=lambda: [])
    logger: Logger = getLogger(__name__)

    def __post_init__(self):
        self.estimators = list(self._get_estimator())

    def _get_estimator(self):
        return []

    def _get_estimator_repr(self, estimator):
        return Estimator._get_estimator_repr(estimator)

    def _get_overrides_text(self, estimator):
        return ""

    def _logger(self, estimator):
        return lambda text: getLogger(type(estimator).__name__).info(text)

    def _step_text(self, step_name, step_number, estimator):
        # step text variables
        step = step_number + 1
        n_steps = len(self.estimators)
        overrides_text = self._get_overrides_text(estimator)
        estimator_repr = self._get_estimator_repr(estimator)

        return lambda secs: (
            TerminalColor.yellow(f"{overrides_text}")
            + f"{estimator_repr} ... {step_name} "
            + "in "
            + TerminalColor.cyan(f"{format_timespan(secs)} ")
            + TerminalColor.green("✓ ")
            + "("
            + TerminalColor.purple(f"step {step}/{n_steps}")
            + ")"
        )

    def _prepare_data(self, X, y):
        return X, y

    def fit(self, X, y) -> AbstractEstimator:
        """Sequentially fits all estimators in this experiment, and record timings;
        which will be stored in a `fit_time_` attribute in each estimator itself.

        Args:
            X (np.ndarray): design matrix X
            y (np.ndarray): target labels y"""

        X, y = self._prepare_data(X, y)

        for step_number, estimator in enumerate(self.estimators):
            logger = self._logger(estimator)
            text = self._step_text("fit", step_number, estimator)

            start_time = perf_counter()
            estimator.fit(X, y)
            fit_time = perf_counter() - start_time
            setattr(estimator, "fit_time_", fit_time)

            logger(text(fit_time))

        return self

    def transform(self, X, y):
        ...

    def fit_transform(self, X, y):
        ...

    def _score_to_dataframe(self, score):
        """Converts a score to a pandas `DataFrame`. If already a DataFrame; returns the
        dataframe, if a scalar; returns a dataframe with one column called 'score' and
        one row representing the scalar."""
        if isinstance(score, pd.DataFrame):
            return score
        elif isinstance(score, float) or isinstance(score, int):
            return pd.DataFrame([{"score": score}])
        else:
            raise ValueError(f"illegal score type received: {type(score)}")

    def score(self, X, y) -> pd.DataFrame:
        """Sequentially scores all estimators in this experiment, and appends the scores
        to a dataframe. Returns all accumulated scores."""
        X, y = self._prepare_data(X, y)
        scores = pd.DataFrame()

        for estimator in self.estimators:
            score = estimator.score(X, y)
            score = self._score_to_dataframe(score)
            scores = scores.append(score)

        return scores
