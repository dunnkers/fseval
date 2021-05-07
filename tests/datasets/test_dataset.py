from dataclasses import dataclass
from typing import Any

import pytest
from fseval.adapters import Adapter
from fseval.datasets import Dataset


@dataclass
class SomeAdapter(Adapter):
    def get_data(self):
        return [[1], [2]], [0, 1]


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
    assert ds.p == 1


def test_params(ds):
    ds.load()
    params = ds.get_params()
    params["n"] == 2
    params["p"] == 1


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
