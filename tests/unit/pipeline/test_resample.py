import numpy as np
import pytest

from fseval.pipeline.resample import Resample


@pytest.fixture
def X():
    return [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]


def test_resample_shuffling(X):
    resampler = Resample(random_state=0)
    X_shuffled = resampler.transform(X)
    assert not (np.array(X) == np.array(X_shuffled)).all()
    assert len(X_shuffled) == 10


def test_sample_size(X):
    resampler = Resample(random_state=0, sample_size=1)
    X_shuffled = resampler.transform(X)
    assert len(X_shuffled) == 1

    resampler = Resample(random_state=0, sample_size=0.2)
    X_shuffled = resampler.transform(X)
    assert len(X_shuffled) == 2

    resampler = Resample(random_state=0, sample_size=1.0)
    X_shuffled = resampler.transform(X)
    assert len(X_shuffled) == 10
