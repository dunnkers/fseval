import inspect
from dataclasses import dataclass
from logging import Logger, getLogger
from typing import Any, Optional

import numpy as np
from fseval.types import AbstractEstimator, Task
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import II, MISSING, DictConfig, OmegaConf
from sklearn.base import BaseEstimator


@dataclass
class EstimatorConfig:
    estimator: Any = None  # must have _target_ of type BaseEstimator.
    # tags:
    multioutput: bool = False
    requires_positive_X: bool = False


@dataclass
class TaskedEstimatorConfig:
    _target_: str = "fseval.pipeline.estimator.instantiate_estimator"
    _recursive_: bool = False  # don't instantiate classifier/regressor
    _target_class_: str = "fseval.pipeline.estimator.Estimator"
    name: str = MISSING
    task: Task = MISSING
    classifier: Optional[EstimatorConfig] = None
    regressor: Optional[EstimatorConfig] = None


@dataclass
class Estimator(AbstractEstimator):
    estimator: Any = MISSING
    logger: Logger = getLogger(__name__)
    name: str = MISSING

    def _get_estimator_repr(self):
        module_path = inspect.getmodule(self.estimator)
        module_name = module_path.__name__
        class_name = type(self.estimator).__name__
        return f"{module_name}.{class_name}"

    def fit(self, X, y):
        self.logger.debug(f"Estimator fit: {self._get_estimator_repr()}")
        self.estimator.fit(X, y)
        return self

    def transform(self, X, y):
        self.logger.debug(f"Estimator transform: {self._get_estimator_repr()}")
        return self.estimator.transform(X, y)

    def fit_transform(self, X, y):
        self.logger.debug(f"Estimator transform plus fit: {self._get_estimator_repr()}")
        return self.fit(X, y).transform(X, y)

    def score(self, X, y):
        self.logger.debug(f"Estimator scoring: {self._get_estimator_repr()}")
        return self.estimator.score(X, y)

    @property
    def feature_importances_(self):
        return self.estimator.feature_importances_


def instantiate_estimator(
    _target_class_: str = MISSING,
    task: Task = MISSING,
    classifier: Optional[EstimatorConfig] = None,
    regressor: Optional[EstimatorConfig] = None,
    **kwargs,
):
    estimator_configs = dict(classification=classifier, regression=regressor)
    estimator_config = estimator_configs[task.name]
    assert (
        estimator_config is not None
    ), f"selected estimator does not support {task.name} datasets!"

    # instantiate estimator
    estimator_config = OmegaConf.to_container(estimator_config)  # type: ignore
    estimator = estimator_config.pop("estimator")  # type: ignore
    estimator = instantiate(estimator)

    # parse and merge tags from estimator
    get_tags = getattr(estimator, "_get_tags", lambda: {})
    more_tags = getattr(estimator, "_more_tags", lambda: {})
    tags = {**get_tags(), **more_tags(), **estimator_config}  # type: ignore
    setattr(estimator, "_get_tags", lambda: tags)

    return instantiate({"_target_": _target_class_, **kwargs}, estimator)
