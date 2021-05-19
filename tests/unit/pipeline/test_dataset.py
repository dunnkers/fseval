from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from fseval.pipeline.dataset import Dataset
from fseval.types import AbstractAdapter


@dataclass
class SomeAdapter(AbstractAdapter):
    def get_data(self):
        return [[1, 3], [2, 4]], [0, 1]


@dataclass
class SomeDataset(Dataset):
    adapter: Any = SomeAdapter()


@pytest.fixture
def ds():
    return SomeDataset()


def test_loading(ds):
    ds.load()

    assert ds.X.ndim == 2
    assert ds.y.ndim == 1
    assert ds.n == 2
    assert ds.p == 2


def test_ensure_loaded(ds):
    with pytest.raises(AssertionError):
        ds._ensure_loaded()


def test_feature_importances(ds):
    ds.load()

    # importance array becomes 1D; uniform distribution is correctly normalized
    ds.feature_importances = {"X[:]": 3.0}
    X_importances = ds.get_feature_importances()
    assert np.isclose(X_importances, [0.5, 0.5]).all()

    # different feature weightings
    ds.feature_importances = {"X[:, 0]": 3.0, "X[:, 1]": 1.0}
    X_importances = ds.get_feature_importances()
    assert np.isclose(X_importances, [0.75, 0.25]).all()

    # instance-based feature importances are also supported
    ds.feature_importances = {"X[0, :]": 1.0, "X[1, 0]": 3, "X[1, 1]": 1}
    X_importances = ds.get_feature_importances()
    assert np.isclose(X_importances, [[0.5, 0.5], [0.75, 0.25]]).all()


def test_callable_adapter_assertion(ds):
    with pytest.raises(AssertionError):
        ds.adapter = lambda: "<incorrect format>"
        ds._get_adapter_data()


def test_callable_adapter(ds):
    ds.adapter = lambda: ([[1]], [1])
    X, y = ds._get_adapter_data()
    assert len(X) == 1
    assert len(y) == 1
