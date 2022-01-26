import inspect
from dataclasses import dataclass
from logging import Logger, getLogger
from time import perf_counter
from typing import Dict, Union

import numpy as np
import pandas as pd
from omegaconf import MISSING

from fseval.config import EstimatorConfig
from fseval.types import (
    AbstractEstimator,
    AbstractStorage,
    CacheUsage,
    IncompatibilityError,
    Task,
)


@dataclass
class Estimator(AbstractEstimator, EstimatorConfig):
    name: str = MISSING
    task: Task = MISSING

    logger: Logger = getLogger(__name__)
    _is_fitted: bool = False

    def __post_init__(self):
        """Perform compatibility checks after initialization. Check whether this
        estimator is compatible with this dataset. Otherwise, raise an error. This error
        can be caught so the pipeline can be exited gracefully."""

        # dataset support: classification/regression
        clf_support: bool = (
            self._estimator_type == "classifier" and self.task == Task.classification
        )
        reg_support: bool = (
            self._estimator_type == "regressor" and self.task == Task.regression
        )
        if not (clf_support or reg_support):
            raise IncompatibilityError(
                f"{self.name} has no estimator defined for {self.task.name} datasets."
            )

        # multioutput support
        if self.is_multioutput_dataset and not self.multioutput:
            raise IncompatibilityError(
                f"dataset is multivariate but {self.name} has no multioutput support."
            )

        # only multioutput: does not support binary targets
        if self.multioutput_only and not self.is_multioutput_dataset:
            raise IncompatibilityError(
                f"{self.name} only works on multioutput datasets."
            )

    @classmethod
    def _get_estimator_repr(cls, estimator):
        name = ""
        if isinstance(estimator, Estimator):
            name = estimator.name
        return f"{name} {type(estimator).__name__}"

    @classmethod
    def _get_class_repr(cls, estimator):
        module_path = inspect.getmodule(estimator)
        module_name = module_path.__name__
        class_name = type(estimator).__name__
        return f"{module_name}.{class_name}"

    def _load_cache(self, filename: str, storage: AbstractStorage):
        if self.load_cache == CacheUsage.never:
            self.logger.debug(
                "cache loading set to `never`: not attempting to load estimator from cache."
            )
            return

        restored = storage.restore_pickle(filename)
        self.estimator = restored or self.estimator
        self._is_fitted = bool(restored)

        if self.load_cache == CacheUsage.must:
            assert self._is_fitted, (
                "Cache usage was set to 'must' but loading cached estimator failed."
                + " Pickle file might be corrupt or could not be found."
            )

    def _save_cache(self, filename: str, storage: AbstractStorage):
        if self.save_cache == CacheUsage.never:
            self.logger.debug("cache saving set to `never`: not caching estimator.")
            return
        else:
            storage.save_pickle(filename, self.estimator)

    def fit(self, X, y):
        # don't refit if cache available and `use_cache_if_available` is enabled
        if self._is_fitted:
            self.logger.debug("using estimator from cache, skipping fit step.")
            return self

        # fit
        self.logger.debug(f"Fitting {Estimator._get_class_repr(self)}...")
        start_time = perf_counter()
        self.estimator.fit(X, y)
        fit_time = perf_counter() - start_time
        self.fit_time_ = fit_time

        return self

    def score(self, X, y, **kwargs) -> Union[Dict, pd.DataFrame, np.generic, None]:
        self.logger.debug(f"Scoring {Estimator._get_class_repr(self)}...")
        return self.estimator.score(X, y)

    @property
    def feature_importances_(self):
        if hasattr(self.estimator, "feature_importances_"):
            return self.estimator.feature_importances_
        elif hasattr(self.estimator, "coef_"):
            return self.estimator.coef_
        else:
            raise ValueError(
                f"no `feature_importances_` found on {Estimator._get_class_repr(self)}"
            )

    @property
    def feature_support_(self):
        return self.estimator.support_

    @property
    def feature_ranking_(self):
        return self.estimator.ranking_

    @property
    def fit_time_(self):
        """ "Retrieves the estimator fitting time. Either the cached fitting time, or
        the time that was just recorded."""
        return self.estimator._fseval_internal_fit_time_

    @fit_time_.setter
    def fit_time_(self, fit_time_):
        """Store the recorded fitting time. Stored in an attribute on the estimator
        itself, so the fitting time is cached inside the estimator object."""
        setattr(self.estimator, "_fseval_internal_fit_time_", fit_time_)
