from hydra.utils import instantiate
from fseval.config import RankerConfig, Task
import pytest
from omegaconf import OmegaConf
import numpy as np
from dataclasses import dataclass


@pytest.fixture(scope="module")
def ranker():
    classifier = dict(_target_="fseval.ranker.chi2.Chi2")
    ranker_cfg = RankerConfig(
        _target_="fseval.ranker.Ranker",
        name="some_ranker",
        classifier=classifier,
        task=Task.classification,
    )
    cfg = OmegaConf.create(ranker_cfg)
    ranker = instantiate(cfg)
    return ranker


def test_initialization(ranker):
    assert isinstance(ranker.name, str)


def test_fit(ranker):
    ranker.fit([[1, 1], [2, 1], [3, 3]], [0, 1, 1])
    assert np.isclose(sum(ranker.feature_importances_), 1)
