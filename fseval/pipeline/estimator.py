import inspect
from dataclasses import dataclass
from logging import Logger, getLogger
from typing import Any, Optional

import numpy as np
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import II, MISSING, DictConfig, OmegaConf
from sklearn.base import BaseEstimator
from sklearn.preprocessing import minmax_scale

from fseval.types import AbstractEstimator, IncompatibilityError, Task


@dataclass
class EstimatorConfig:
    estimator: Any = None  # must have _target_ of type BaseEstimator.
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
    # tags
    multioutput: Optional[bool] = False
    multioutput_only: Optional[bool] = False
    requires_positive_X: Optional[bool] = False
    estimates_feature_importances: Optional[bool] = False  # returns importance scores
    estimates_feature_support: Optional[bool] = False  # returns feature subset
    estimates_feature_ranking: Optional[bool] = False  # returns feature ranking
    estimates_target: Optional[bool] = False  # can predict target
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

    def fit(self, X, y):
        if self.requires_positive_X:
            X = minmax_scale(X)
            self.logger.info(
                "rescaled X: this estimator strictly requires positive features."
            )

        self.logger.debug(f"Fitting {Estimator._get_class_repr(self)}...")
        self.estimator.fit(X, y)
        return self

    def transform(self, X, y):
        self.logger.debug(f"Using {Estimator._get_class_repr(self)} transform...")
        return self.estimator.transform(X, y)

    def fit_transform(self, X, y):
        self.logger.debug(
            f"Fitting and transforming {Estimator._get_class_repr(self)}..."
        )
        return self.fit(X, y).transform(X, y)

    def score(self, X, y):
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
        return self.estimator._fseval_internal_fit_time_

    @fit_time_.setter
    def fit_time_(self, fit_time_):
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
