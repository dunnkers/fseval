import inspect
from dataclasses import dataclass
from logging import Logger, getLogger
from time import perf_counter
from typing import Any, Optional

from fseval.types import (
    AbstractEstimator,
    AbstractStorageProvider,
    CacheUsage,
    IncompatibilityError,
    Task,
)
from hydra.utils import instantiate
from omegaconf import II, MISSING, OmegaConf
from sklearn.preprocessing import minmax_scale


@dataclass
class EstimatorConfig:
    estimator: Any = None  # must have _target_ of type BaseEstimator.
    load_cache: Optional[CacheUsage] = None
    save_cache: Optional[CacheUsage] = None
    # tags
    multioutput: Optional[bool] = None
    multioutput_only: Optional[bool] = None
    requires_positive_X: Optional[bool] = None
    estimates_feature_importances: Optional[bool] = None
    estimates_feature_support: Optional[bool] = None
    estimates_feature_ranking: Optional[bool] = None
    estimates_target: Optional[bool] = None


@dataclass
class TaskedEstimatorConfig(EstimatorConfig):
    _target_: str = "fseval.pipeline.estimator.instantiate_estimator"
    _recursive_: bool = False  # don't instantiate classifier/regressor
    _target_class_: str = "fseval.pipeline.estimator.Estimator"
    name: str = MISSING
    classifier: Optional[EstimatorConfig] = None
    regressor: Optional[EstimatorConfig] = None
    load_cache: CacheUsage = CacheUsage.allow
    save_cache: CacheUsage = CacheUsage.allow
    # tags
    multioutput: bool = False
    multioutput_only: bool = False
    requires_positive_X: bool = False
    estimates_feature_importances: bool = False
    estimates_feature_support: bool = False
    estimates_feature_ranking: bool = False
    estimates_target: bool = False
    # runtime properties
    task: Task = II("dataset.task")
    is_multioutput_dataset: bool = II("dataset.multioutput")


@dataclass
class Estimator(AbstractEstimator, EstimatorConfig):
    # make sure `estimator` is always the first property: we pass it as a positional
    # argument in `instantiate_estimator`.
    name: str = MISSING
    task: Task = MISSING

    logger: Logger = getLogger(__name__)
    _is_fitted: bool = False

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

    def _load_cache(self, filename: str, storage_provider: AbstractStorageProvider):
        if self.load_cache == CacheUsage.never:
            self.logger.debug(
                "cache loading set to `never`: not attempting to load estimator from cache."
            )
            return

        restored = storage_provider.restore_pickle(filename)
        self.estimator = restored or self.estimator
        self._is_fitted = bool(restored)

        if self.load_cache == CacheUsage.must:
            assert self._is_fitted, (
                "Cache usage was set to 'must' but loading cached estimator failed."
                + " Pickle file might be corrupt or could not be found."
            )

    def _save_cache(self, filename: str, storage_provider: AbstractStorageProvider):
        if self.save_cache == CacheUsage.never:
            self.logger.debug("cache saving set to `never`: not caching estimator.")
            return
        else:
            storage_provider.save_pickle(filename, self.estimator)
            # TODO check whether file was successfully saved.

    def fit(self, X, y):
        # don't refit if cache available and `use_cache_if_available` is enabled
        if self._is_fitted:
            self.logger.debug("using estimator from cache, skipping fit step.")
            return self

        # rescale if necessary
        if self.requires_positive_X:
            X = minmax_scale(X)
            self.logger.info(
                "rescaled X: this estimator strictly requires positive features."
            )

        # fit
        self.logger.debug(f"Fitting {Estimator._get_class_repr(self)}...")
        start_time = perf_counter()
        self.estimator.fit(X, y)
        fit_time = perf_counter() - start_time
        self.fit_time_ = fit_time

        return self

    def transform(self, X, y):
        self.logger.debug(f"Using {Estimator._get_class_repr(self)} transform...")
        return self.estimator.transform(X, y)

    def fit_transform(self, X, y):
        self.logger.debug(
            f"Fitting and transforming {Estimator._get_class_repr(self)}..."
        )
        return self.fit(X, y).transform(X, y)

    def score(self, X, y, **kwargs):
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


def instantiate_estimator(
    _target_class_: str = MISSING,
    name: str = MISSING,
    task: Task = MISSING,
    is_multioutput_dataset: bool = MISSING,
    estimator: Any = None,
    classifier: Optional[EstimatorConfig] = None,
    regressor: Optional[EstimatorConfig] = None,
    **top_level_tags,
):
    estimator_configs = dict(classification=classifier, regression=regressor)
    estimator_config = estimator_configs[task.name] or estimator

    # dataset support: classification/regression
    if estimator_config is None:
        raise IncompatibilityError(
            f"{name} has no estimator defined for {task.name} datasets."
        )

    # pop `estimator` out - all that's left should be tags.
    estimator_config = OmegaConf.to_container(estimator_config)  # type: ignore
    estimator = estimator_config.pop("estimator")  # type: ignore

    # tags. map any tag override from task-estimator to the top-level estimator
    estimator_tags = {k: v for k, v in estimator_config.items() if v is not None}  # type: ignore
    tags = {**top_level_tags, **estimator_tags}  # type: ignore

    # multioutput support
    if is_multioutput_dataset and not tags["multioutput"]:
        raise IncompatibilityError(
            f"dataset is multivariate but {name} has no multioutput support."
        )

    # only multioutput: does not support binary targets
    if tags["multioutput_only"] and not is_multioutput_dataset:
        raise IncompatibilityError(f"{name} only works on multioutput datasets.")

    # instantiate estimator and its wrapping estimator
    estimator = instantiate(estimator)
    instance = instantiate(
        {"_target_": _target_class_, **tags}, estimator, name=name, task=task
    )

    return instance
