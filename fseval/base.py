import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, cast

import numpy as np
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig, OmegaConf
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, r2_score


class Task(Enum):
    regression = 1
    classification = 2


class Configurable(BaseEstimator):
    @classmethod
    def _get_config_names(cls) -> List:
        """Get config names for this configurable estimator"""
        return super()._get_param_names()

    def _omitted_values(self) -> List:
        return [MISSING, None]

    def get_config(self, deep: bool = True) -> Dict:
        keys = self._get_config_names()
        values = [getattr(self, key) for key in keys]
        out = dict()
        for key, value in zip(keys, values):
            if value in self._omitted_values():
                continue
            if deep and hasattr(value, "get_config"):
                out[key] = value.get_config()
            elif deep and hasattr(value, "get_params"):
                out[key] = value.get_params()
            elif isinstance(value, Enum):
                out[key] = value.name
            # TODO recurse with dict keys/values
            elif isinstance(value, DictConfig):
                out[key] = OmegaConf.to_container(value)
            elif hasattr(value, "__dict__"):
                out[key] = value.__dict__
            else:
                out[key] = value

        return out
