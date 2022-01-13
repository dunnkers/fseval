import logging
from typing import Optional

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import resample

from fseval.config import ResampleConfig

logger = logging.getLogger(__name__)


class Resample(ResampleConfig, BaseEstimator, TransformerMixin):
    n_samples: Optional[int] = None
    frac_samples: Optional[float] = None

    def fit(self, *arrays, y=None):
        return self

    def transform(self, *arrays):
        # assume arrays have equal amount of samples. `resample` also checks.
        n = len(arrays[0])

        if isinstance(self.sample_size, int):
            self.n_samples = self.sample_size
        elif isinstance(self.sample_size, float):
            self.n_samples = round(n * self.sample_size)
        else:  # use all samples when no `sample_size` given
            self.n_samples = n
        self.frac_samples = self.n_samples / n

        samples = resample(
            *arrays,
            replace=self.replace,
            random_state=self.random_state,
            stratify=self.stratify,
            n_samples=self.n_samples,
        )

        with_or_without = "with" if self.replace else "without"
        logger.info(
            f"took {self.n_samples} samples {with_or_without} replacement"
            + f" (total samples: n={n}, random_state={self.random_state})"
        )

        return samples
