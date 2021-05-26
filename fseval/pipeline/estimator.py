import inspect
from dataclasses import dataclass
from logging import Logger, getLogger
from typing import Any, Optional

import numpy as np
from fseval.types import AbstractEstimator, IncompatibilityError, Task
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import II, MISSING, DictConfig, OmegaConf
from sklearn.base import BaseEstimator
from sklearn.preprocessing import minmax_scale


@dataclass
class EstimatorConfig:
    estimator: Any = None  # must have _target_ of type BaseEstimator.
    # tags:
    multioutput: bool = False
    multioutput_only: bool = False
    requires_positive_X: bool = False


@dataclass
class TaskedEstimatorConfig:
    _target_: str = "fseval.pipeline.estimator.instantiate_estimator"
    _recursive_: bool = False  # don't instantiate classifier/regressor
    _target_class_: str = "fseval.pipeline.estimator.Estimator"
    name: str = MISSING
    task: Task = II("dataset.task")
    classifier: Optional[EstimatorConfig] = None
    regressor: Optional[EstimatorConfig] = None
    # dataset runtime properties
    is_multioutput_dataset: bool = II("dataset.multioutput")


@dataclass
class Estimator(AbstractEstimator):
    estimator: Any = MISSING
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
        if self._get_tags().get("requires_positive_X"):
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
        return self.estimator.feature_importances_

    @property
    def fit_time_(self):
        return self.estimator._fseval_internal_fit_time_

    @fit_time_.setter
    def fit_time_(self, fit_time_):
        setattr(self.estimator, "_fseval_internal_fit_time_", fit_time_)

    def _get_tags(self):
        return self.estimator._get_tags()


def instantiate_estimator(
    _target_class_: str = MISSING,
    name: str = MISSING,
    task: Task = MISSING,
    is_multioutput_dataset: bool = MISSING,
    classifier: Optional[EstimatorConfig] = None,
    regressor: Optional[EstimatorConfig] = None,
    **kwargs,
):
    estimator_configs = dict(classification=classifier, regression=regressor)
    estimator_config = estimator_configs[task.name]

    # raise incompatibility error if no estimator found for this dataset type
    if estimator_config is None:
        raise IncompatibilityError(
            f"{name} has no estimator defined for {task.name} datasets."
        )

    if is_multioutput_dataset and not estimator_config.multioutput:
        raise IncompatibilityError(
            f"dataset is multivariate but {name} has no multioutput support."
        )

    if estimator_config.multioutput_only and not is_multioutput_dataset:
        raise IncompatibilityError(f"{name} only works on multioutput datasets.")

    # instantiate estimator
    estimator_config = OmegaConf.to_container(estimator_config)  # type: ignore
    estimator = estimator_config.pop("estimator")  # type: ignore
    estimator = instantiate(estimator)

    # parse and merge tags from estimator
    get_tags = getattr(estimator, "_get_tags", lambda: {})
    more_tags = getattr(estimator, "_more_tags", lambda: {})
    tags = {**get_tags(), **more_tags(), **estimator_config}  # type: ignore
    setattr(estimator, "_get_tags", lambda: tags)

    instance = instantiate(
        {"_target_": _target_class_, "name": name, **kwargs}, estimator, task=task
    )
    return instance
