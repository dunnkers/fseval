from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from fseval.pipeline.dataset import Dataset, DatasetLoader
from fseval.types import AbstractAdapter, Task
from omegaconf import OmegaConf


@dataclass
class SomeAdapter(AbstractAdapter):
    def get_data(self):
        return [[1, 3], [2, 4]], [0, 1]


@dataclass
class SomeDataset(DatasetLoader):
    task: Task = Task.classification
    adapter: Any = SomeAdapter()


@pytest.fixture
def ds_loader():
    return SomeDataset()


def test_loading(ds_loader):
    ds: Dataset = ds_loader.load()

    assert ds.X.ndim == 2
    assert ds.y.ndim == 1
    assert ds.n == 2
    assert ds.p == 2
    assert isinstance(ds._log_details, str)


def test_feature_importances(ds_loader):
    # importance array becomes 1D; uniform distribution is correctly normalized
    ds_loader.feature_importances = {"X[:]": 3.0}
    ds: Dataset = ds_loader.load()
    assert np.isclose(ds.feature_importances, [0.5, 0.5]).all()

    # different feature weightings
    ds_loader.feature_importances = {"X[:, 0]": 3.0, "X[:, 1]": 1.0}
    ds: Dataset = ds_loader.load()
    assert np.isclose(ds.feature_importances, [0.75, 0.25]).all()

    # instance-based feature importances are also supported
    ds_loader.feature_importances = {"X[0, :]": 1.0, "X[1, 0]": 3, "X[1, 1]": 1}
    ds: Dataset = ds_loader.load()
    assert np.isclose(ds.feature_importances, [[0.5, 0.5], [0.75, 0.25]]).all()


def test_callable_adapter_assertion(ds_loader):
    with pytest.raises(AssertionError):
        ds_loader.adapter = lambda: "<incorrect format>"
        ds_loader._get_adapter_data()


def test_get_adapter_omegadict(ds_loader):
    ds_loader.adapter = OmegaConf.create(
        {"_target_": "tests.unit.pipeline.test_dataset.SomeAdapter"}
    )
    adapter: AbstractAdapter = ds_loader._get_adapter()
    assert isinstance(adapter, SomeAdapter)


def test_get_adapter_wrong_value(ds_loader):
    """Setting the adapter to a float should definitely fail."""
    with pytest.raises(AssertionError):
        ds_loader.adapter = 1.0
        ds_loader._get_adapter_data()


def test_get_adapter_dict_value(ds_loader):
    ds_loader.adapter = {}
    adapter = ds_loader._get_adapter()
    assert adapter == {}


def test_callable_adapter(ds_loader):
    ds_loader.adapter = lambda: ([[1]], [1])
    X, y = ds_loader._get_adapter_data()
    assert len(X) == 1
    assert len(y) == 1
