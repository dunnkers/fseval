from dataclasses import dataclass
from fseval.config import ResampleConfig
from sklearn.utils import resample
from sklearn.base import TransformerMixin
from fseval.base import Configurable


class Resample(ResampleConfig, TransformerMixin, Configurable):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        kwargs = self.get_params()
        kwargs.pop("_target_")
        return resample(X, **kwargs)
