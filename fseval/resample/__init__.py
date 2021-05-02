from dataclasses import dataclass
from fseval.config import ResampleConfig
from sklearn.utils import resample
from sklearn.base import TransformerMixin, BaseEstimator


class Resample(ResampleConfig, TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return resample(X, **self.get_params())
