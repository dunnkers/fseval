from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from fseval.adapters import Adapter
from fseval.datasets import Dataset


@dataclass
class SomeAdapter(Adapter):
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


def test_params(ds):
    ds.load()
    params = ds.get_params()
    params["n"] == 2
    params["p"] == 2


def test_ensure_loaded(ds):
    with pytest.raises(AssertionError):
        ds._ensure_loaded()


def test_get_subsets(ds):
    ds.load()
    subsets = ds.get_subsets([0], [1])
    X_train, X_test, y_train, y_test = subsets

    assert len(subsets) == 4
    assert X_train.shape[0] == 1
    assert X_test.shape[0] == 1
    assert y_train.shape[0] == 1
    assert y_test.shape[0] == 1


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
