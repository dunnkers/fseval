import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, cast

import numpy as np
from fseval.base import Task
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import II, MISSING, DictConfig, OmegaConf
from sklearn.base import BaseEstimator


@dataclass
class EstimatorConfig:
    name: str = MISSING
    estimator: Any = None  # must have _target_ of type BaseEstimator.
    multioutput: bool = False
    requires_positive_X: bool = False


@dataclass
class TaskedEstimatorConfig:
    _target_: str = "fseval.pipeline.estimator.instantiate_estimator"
    _recursive_: bool = False  # don't instantiate classifier/regressor
    name: str = MISSING
    task: Task = MISSING
    classifier: Optional[EstimatorConfig] = None
    regressor: Optional[EstimatorConfig] = None


def instantiate_estimator(
    task: Task,
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

    # create a meaningful config object
    def get_config():
        params = estimator.get_params()
        config = {"params": params, **estimator_config, **kwargs}
        return config

    setattr(estimator, "get_config", get_config)

    return estimator
