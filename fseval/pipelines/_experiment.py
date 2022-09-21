import multiprocessing
from dataclasses import dataclass, field
from functools import reduce
from logging import Logger, getLogger
from time import perf_counter
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from humanfriendly import format_timespan

from fseval.pipeline.estimator import Estimator
from fseval.types import AbstractEstimator, Callback, TerminalColor


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
        # logs
        logger = self._logger(estimator)
        text = self._step_text("fit", step_number, estimator)

        # fit & print time elapsed
        start_time = perf_counter()
        estimator.fit(X, y)
        fit_time = perf_counter() - start_time
        logger(text(fit_time))

        return estimator

    def _remove_and_get_callbacks(self) -> Tuple[Dict[str, Callback], List[str]]:
        """
        Removes callbacks from this Experiment object. Returns them as a
        tuple. This is necessary because the callbacks might contain state that
        either **cannot** be pickled, or **should** not be taken over to a
        process fork. For example, SQLAlchemy's engine object can intentionally
        not be forked - and with good reason.

        @see https://docs.sqlalchemy.org/en/14/core/pooling.html
        """
        callback_objects = dict()
        callback_names = self.callbacks.callback_names

        for callback_name in self.callbacks.callback_names:
            callback_objects[callback_name] = getattr(self.callbacks, callback_name)
            delattr(self.callbacks, callback_name)

        self.callbacks.callback_names = []

        return callback_objects, callback_names

    def _set_callbacks(
        self, callback_objects: Dict[str, Callback], callback_names: List[str]
    ):
        """
        Restores the callbacks, by setting them back onto this class object.

        Attributes:
            callback_objects (Dict[str, Callback]): The callback objects, like stored in
                `_remove_and_get_callbacks()`.
            callback_names (List[str]): The corresponding callback names, like stored in
                `_remove_and_get_callbacks()`.
        """
        self.callbacks.callback_names = callback_names

        for callback_name in self.callbacks.callback_names:
            setattr(self.callbacks, callback_name, callback_objects[callback_name])

    def fit(self, X, y) -> AbstractEstimator:
        """Sequentially fits all estimators in this experiment.

        Attributes:
            X (np.ndarray): design matrix X
            y (np.ndarray): target labels y"""

        X, y = self._prepare_data(X, y)

        ## Run `fit`
        n_jobs = self._get_n_jobs()
        if n_jobs is not None and (n_jobs > 1 or n_jobs == -1):
            assert n_jobs >= 1 or n_jobs == -1, f"incorrect `n_jobs`: {n_jobs}"

            # remove callbacks this object and store locally in main thread.
            callback_objects, callback_names = self._remove_and_get_callbacks()

            # determine amount of CPU's to use. ALL if n_jobs is -1, else n_jobs.
            cpus = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
            self.logger.info(f"Using {cpus} CPU's in parallel (n_jobs={n_jobs})")

            # input to `self._fit_esitmator`
            star_input = [
                (X, y, step_number, estimator)
                for step_number, estimator in enumerate(self.estimators)
            ]

            # open pool and fit estimators.
            pool = multiprocessing.Pool(processes=cpus)
            estimators = pool.starmap(self._fit_estimator, star_input)
            pool.close()
            pool.join()

            # restore callbacks in main thread
            self._set_callbacks(callback_objects, callback_names)

            # set collected estimators to this local object
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

    def _aggregate_dataframe_scores(self, a: pd.DataFrame, b: pd.DataFrame):
        return pd.concat([a, b])

    def _aggregate_dict_scores(self, a: Dict, b: Dict):
        aggregated = {}

        # merge common keys
        common_keys_filter = filter(lambda key: key in b, a)
        common_keys = list(common_keys_filter)

        for key in common_keys:
            value_a = a.pop(key)
            value_b = b.pop(key)
            aggregated[key] = self._aggregate_scores(value_a, value_b)

        # add remaining uncommon keys
        aggregated = {**aggregated, **a, **b}

        return aggregated

    def _aggregate_scores(
        self, a: Union[pd.DataFrame, Dict], b: Union[pd.DataFrame, Dict]
    ):
        is_dataframe = isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame)
        is_dict = isinstance(a, Dict) and isinstance(b, Dict)

        if is_dataframe:
            agg_scores = self._aggregate_dataframe_scores(a, b)
            return agg_scores
        elif is_dict:
            agg_scores = self._aggregate_dict_scores(a, b)
            return agg_scores
        else:
            raise ValueError(
                "In an `Experiment`, all estimators must either return only DataFrames "
                + "or Dictionaries containing DataFrames. Otherwise, the experiment "
                + "results cannot be aggregated. Got: "
                + f"{type(a)} and {type(b)}."
            )

    def score(self, X, y, **kwargs) -> Union[Dict, pd.DataFrame, np.generic, None]:
        """Sequentially scores all estimators in this experiment, and appends the scores
        to a dataframe or a dict containing dataframes. Returns all accumulated scores.
        """
        X, y = self._prepare_data(X, y)

        # score all estimators and aggregate
        scores = [estimator.score(X, y, **kwargs) for estimator in self.estimators]
        scores_agg = reduce(self._aggregate_scores, scores)

        return scores_agg
