from hydra.utils import instantiate
from fseval.config import RankerConfig, Task
import pytest
from omegaconf import OmegaConf, MISSING
import numpy as np
from dataclasses import dataclass
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
    assert ranker._target_ == MISSING

    config = ranker.get_config()
    assert "estimator" in config
    assert "_target_" not in config  # removed because value is missing
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