from dataclasses import dataclass

import numpy as np
import pytest
from fseval.config import RankerConfig, Task
from hydra.utils import instantiate
from omegaconf import MISSING, OmegaConf
from sklearn.base import clone


@pytest.fixture(scope="module")
def ranker():
    classifier = dict(_target_="fseval.rankers.Chi2")
    ranker_cfg = RankerConfig(
        _target_="fseval.rankers.Ranker",
        name="some_ranker",
        classifier=classifier,
        task=Task.classification,
    )
    cfg = OmegaConf.create(ranker_cfg)
    ranker = instantiate(cfg)
    return ranker


def test_initialization(ranker):
    assert isinstance(ranker.name, str)

    config = ranker.get_config()
    assert "estimator" in config
    # should recursively get_params from other BaseEstimator or Configurable's
    assert isinstance(config["estimator"], dict)


def test_fit(ranker):
    ranker.fit([[1, 1], [2, 1], [3, 3]], [0, 1, 1])
    assert np.isclose(sum(ranker.feature_importances_), 1)
    relevant_features = [0, 1]
    assert ranker.score(None, [0, 1]) > 0


def test_can_clone(ranker):
    cloned_ranker = clone(ranker)
    assert cloned_ranker is not None
