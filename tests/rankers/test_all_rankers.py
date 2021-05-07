import pytest
from fseval.config import Task
from fseval.rankers import Ranker
from hydra.utils import instantiate
from tests.utils import get_single_config

ALL_RANKERS = ["chi2", "relieff", "tabnet"]


@pytest.fixture(params=ALL_RANKERS)
def ranker_cfg(config_repo, request):
    single_config = get_single_config(config_repo, "ranker", request.param)
    return single_config


def test_initialization(ranker_cfg):
    ranker_cfg["task"] = Task.classification
    ranker = instantiate(ranker_cfg)
    assert isinstance(ranker, Ranker)

    ranker_cfg["task"] = Task.regression
    ranker = instantiate(ranker_cfg)
    assert isinstance(ranker, Ranker)
