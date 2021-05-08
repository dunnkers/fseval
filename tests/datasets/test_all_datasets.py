import numpy as np
import pytest
from fseval.datasets import Dataset
from hydra.utils import instantiate
from tests.utils import get_single_config

ALL_DATASETS = ["boston", "iris", "switch", "xor"]


@pytest.fixture(params=ALL_DATASETS)
def dataset_cfg(config_repo, request):
    single_config = get_single_config(config_repo, "dataset", request.param)
    return single_config


def test_initialization(dataset_cfg):
    ds = instantiate(dataset_cfg)
    assert isinstance(ds, Dataset)


def test_feature_relevancy(dataset_cfg):
    ds = instantiate(dataset_cfg)
    ds.load()
    if not ds.feature_relevancy:
        return
    else:
        relevances = ds.relevant_features
        assert relevances is not None
        assert relevances.shape == ds.X.shape
