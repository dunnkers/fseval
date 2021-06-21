import multiprocessing
from dataclasses import dataclass, field
from logging import Logger, getLogger
from time import perf_counter
from typing import List

import pandas as pd
from fseval.pipeline.estimator import Estimator
from fseval.types import AbstractEstimator, TerminalColor
from humanfriendly import format_timespan


@dataclass
class Experiment(AbstractEstimator):
    estimators: List[AbstractEstimator] = field(default_factory=lambda: [])
    logger: Logger = getLogger(__name__)

    def __post_init__(self):
        self.estimators = list(self._get_estimator())

    def _get_n_jobs(self):
        return None

    def _get_estimator(self):
        return []

    def _get_estimator_repr(self, estimator):
        return Estimator._get_estimator_repr(estimator)

    def _get_overrides_text(self, estimator):
        return ""

    def _logger(self, estimator):
        return lambda text: getLogger(type(estimator).__name__).info(text)

    def _step_text(self, step_name, step_number, estimator):
        """Provides a console logging string for logging during an experiment phase,
        like in `fit` or `score`. Adds coloring and fit times to stdout."""

        # step text variables
        step = step_number + 1
        n_steps = len(self.estimators)
        overrides_text = self._get_overrides_text(estimator)
        estimator_repr = self._get_estimator_repr(estimator)

        return lambda secs: (
            overrides_text
            + TerminalColor.yellow(f"{estimator_repr}")
            + f" ... {step_name} "
            + "in "
            + TerminalColor.cyan(f"{format_timespan(secs)} ")
            + TerminalColor.green("✓ ")
            + "("
            + TerminalColor.purple(f"step {step}/{n_steps}")
            + ")"
        )

    def _prepare_data(self, X, y):
        """Callback. Can be used to implement any data preparation schemes."""
        return X, y

    def prefit(self):
        """Pre-fit hook. Is executed right before calling `fit()`. Can be used to load
        estimators from cache or do any other preparatory work."""

        for estimator in self.estimators:
            if hasattr(estimator, "prefit") and callable(getattr(estimator, "prefit")):
                estimator.prefit()

    def _fit_estimator(self, X, y, step_number, estimator):
        logger = self._logger(estimator)
        text = self._step_text("fit", step_number, estimator)

        # fit & print time elapsed
        start_time = perf_counter()
        estimator.fit(X, y)
        fit_time = perf_counter() - start_time
        logger(text(fit_time))

        return estimator

    def fit(self, X, y) -> AbstractEstimator:
        """Sequentially fits all estimators in this experiment.

        Args:
            X (np.ndarray): design matrix X
            y (np.ndarray): target labels y"""

        X, y = self._prepare_data(X, y)

        ## Run `fit`
        n_jobs = self._get_n_jobs()
        if n_jobs is not None and (n_jobs > 1 or n_jobs == -1):
            assert n_jobs >= 1 or n_jobs == -1, f"incorrect `n_jobs`: {n_jobs}"

            cpus = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
            self.logger.info(f"Using {cpus} CPU's in parallel (n_jobs={n_jobs})")

            star_input = [
                (X, y, step_number, estimator)
                for step_number, estimator in enumerate(self.estimators)
            ]

            pool = multiprocessing.Pool(processes=cpus)
            estimators = pool.starmap(self._fit_estimator, star_input)
            pool.close()
            pool.join()

            self.estimators = estimators
        else:
            for step_number, estimator in enumerate(self.estimators):
                self._fit_estimator(X, y, step_number, estimator)

        return self

    def postfit(self):
        """Post-fit hook. Is executed right after calling `fit()`. Can be used to save
        estimators to cache, for example."""

        for estimator in self.estimators:
            if hasattr(estimator, "postfit") and callable(
                getattr(estimator, "postfit")
            ):
                estimator.postfit()

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

    def score(self, X, y, **kwargs) -> pd.DataFrame:
        """Sequentially scores all estimators in this experiment, and appends the scores
        to a dataframe. Returns all accumulated scores."""
        X, y = self._prepare_data(X, y)
        scores = pd.DataFrame()

        for estimator in self.estimators:
            score = estimator.score(X, y, **kwargs)
            score = self._score_to_dataframe(score)
            scores = scores.append(score)

        return scores
