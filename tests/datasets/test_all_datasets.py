import re

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
    if not ds.feature_relevancy:
        return

    # ground truth
    gt = ds.feature_relevancy
    for obj in ds.feature_relevancy:  # e.g. { 'X[:]': 1.0 }
        for key, value in obj.items():  # e.g. key='X[:]' and value=1.0
            match = re.search("X\[(.*)\]", key)
            assert match is not None
            selector = match.group(1)
            pass
