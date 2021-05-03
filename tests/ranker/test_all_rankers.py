from hydra.utils import instantiate
from fseval.config import RankerConfig, Task
import pytest
from omegaconf import OmegaConf
import numpy as np
from dataclasses import dataclass


# @pytest.fixture(scope="module", params=["fseval.ranker.chi2.Chi2", "skrebate.ReliefF"])
# def ranker_target(request):
#     return request.param


@pytest.fixture(scope="module")
def ranker():
    ranker_cfg = RankerConfig(
        _target_="fseval.ranker.Ranker",
        name="some_ranker",
        compatibility=["multiclass", "multivariate"],
        task=Task.classification,
    )
    cfg = OmegaConf.create(ranker_cfg)
    ranker = instantiate(cfg)
    return ranker


def test_initialization(ranker):
    assert isinstance(ranker.name, str)
    assert len(ranker.compatibility) == 2


def test_fit(ranker):
    """
    no classifier or regressor defined, so should throw error when trying to fit.
    """
    with pytest.raises(AssertionError):
        ranker.fit([[1]], [0])
