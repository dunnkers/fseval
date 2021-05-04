from sklearn.base import BaseEstimator
from typing import Dict, Any
from omegaconf import MISSING


class Configurable(BaseEstimator):
    @classmethod
    def _get_config_names(cls):
        """Get config names for this configurable estimator"""
        return super()._get_param_names()

    def _omitted_values(self):
        return [MISSING]

    def get_config(self, deep: bool = True):
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
            else:
                out[key] = value

        return out
