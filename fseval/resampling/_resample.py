import logging
from typing import Optional

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.utils import resample

from fseval.base import Configurable
from fseval.config import ResampleConfig

logger = logging.getLogger(__name__)


class Resample(ResampleConfig, TransformerMixin, Configurable):
    n_samples: Optional[int] = None
    frac_samples: Optional[float] = None

    @classmethod
    def _get_config_names(cls):
        config = super()._get_config_names()
        config.remove("sample_size")
        config.append("n_samples")
        config.append("frac_samples")
        return config

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert (
            np.ndim(X) >= 1
        ), "tried to resample X, but given array has no dimensions."
        n = np.shape(X)[0]
        if isinstance(self.sample_size, int):
            self.n_samples = self.sample_size
        elif isinstance(self.sample_size, float):
            self.n_samples = round(n * self.sample_size)
        else:  # use all samples when no `sample_size` given
            self.n_samples = n
        self.frac_samples = self.n_samples / n

        samples = resample(
            X,
            replace=self.replace,
            random_state=self.random_state,
            stratify=self.stratify,
            n_samples=self.n_samples,
        )

        with_or_without = "with" if self.replace else "without"
        logger.info(
            f"took {self.n_samples} samples {with_or_without} replacement"
            + f" (total samples: n={n})"
        )

        return samples
