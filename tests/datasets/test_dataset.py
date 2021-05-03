from fseval.datasets import Dataset
from fseval.config import DatasetConfig, Task
import numpy as np
import pytest
from omegaconf import OmegaConf
from hydra.errors import HydraException
from hydra.utils import instantiate


@pytest.fixture
def cfg():
    ds_cfg = DatasetConfig(
        _target_="fseval.datasets.Dataset",
        name="some_ds",
        task=Task.classification,
    )
    cfg = OmegaConf.create(ds_cfg)
    return cfg


def test_no_adapter(cfg):
    with pytest.raises(HydraException):
        instantiate(cfg)  # no adapter configured yet


def test_instantiate(cfg):
    adapter = dict(
        _target_="fseval.adapters.OpenML", dataset_id=531, target_column="MEDV"
    )
    cfg.adapter = adapter
    ds = instantiate(cfg)
    assert isinstance(ds, Dataset)


# class SomeAdapter:
#     def get_data(self):
#         return [[1], [2]], [0, 1]


# class SomeDataset(Dataset):
#     @property
#     def adapter(self):
#         return SomeAdapter()


# @pytest.fixture
# def ds():
#     return SomeDataset()


# def test_loading(ds):
#     ds.load()

#     assert ds.X.ndim == 2
#     assert ds.y.ndim == 1
#     assert ds.n == 2
#     assert ds.p == 1


# def test_params(ds):
#     ds.load()
#     params = ds.get_params()
#     params["n"] == 2
#     params["p"] == 1


# def test_ensure_loaded(ds):
#     with pytest.raises(AssertionError):
#         ds._ensure_loaded()


# def test_get_subsets(ds):
#     ds.load()
#     subsets = ds.get_subsets([0], [1])
#     X_train, X_test, y_train, y_test = subsets

#     assert len(subsets) == 4
#     assert X_train.shape[0] == 1
#     assert X_test.shape[0] == 1
#     assert y_train.shape[0] == 1
#     assert y_test.shape[0] == 1
