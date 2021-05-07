import pytest
from fseval.datasets import Dataset
from hydra.utils import instantiate

ALL_DATASETS = ["boston", "iris", "switch", "xor"]


# @pytest.mark.usefixtures("get_single_config")
@pytest.fixture(params=ALL_DATASETS)
def cfg(get_single_config, request):
    cfg = get_single_config("dataset", request.param)
    return cfg


def test_initialization(cfg):
    ds = instantiate(cfg)
    assert isinstance(ds, Dataset)
