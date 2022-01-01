import tempfile
from typing import cast

import pytest
from fseval.config import EstimatorConfig
from fseval.pipeline.estimator import Estimator
from fseval.storage.local import LocalStorage
from fseval.types import CacheUsage, Task
from hydra.utils import instantiate
from omegaconf import OmegaConf


@pytest.fixture
def estimator_cfg():
    estimator_config = EstimatorConfig(
        name="some_estimator",
        estimator={"_target_": "sklearn.tree.DecisionTreeClassifier"},
        task=Task.classification,
        _estimator_type="classifier",
        is_multioutput_dataset=False,
    )
    estimator_cfg = OmegaConf.create(estimator_config.__dict__)

    return estimator_cfg


def test_estimator_initiation(estimator_cfg: EstimatorConfig):
    estimator = instantiate(estimator_cfg)

    # verify attributes
    assert hasattr(estimator, "estimator")
    assert estimator.name == "some_estimator"
    assert estimator.logger is not None

    # verify feature_importances_
    estimator.fit([[1, 2]], [0])
    assert hasattr(estimator, "feature_importances_")

    # verify scoring
    score = estimator.score([[1, 2]], [0])
    assert score >= 0


def test_estimator_cache(estimator_cfg: EstimatorConfig):
    estimator_cfg.load_cache = CacheUsage.must
    estimator: Estimator = instantiate(estimator_cfg)

    # initiate storage
    tmpdir: str = tempfile.mkdtemp()
    storage: LocalStorage = LocalStorage(load_dir=tmpdir, save_dir=tmpdir)
    filename: str = "fit_estimator.pickle"

    # fit and store to cache
    X, y = [[1, 2]], [0]
    estimator.fit(X, y)
    estimator._save_cache(filename, storage)

    # retrieve from cache
    new_estimator: Estimator = instantiate(estimator_cfg)
    new_estimator._load_cache(filename, storage)

    # verify whether fitted. raises `sklearn.exceptions.NotFittedError`` if not
    # loading from cache was unsuccessful.
    score = new_estimator.score(X, y)
    score = cast(float, score)
    assert score >= 0

    # should not refit
    assert new_estimator._is_fitted
    old_fit_time = new_estimator.fit_time_
    new_estimator.fit(X, y)
    new_fit_time = new_estimator.fit_time_
    assert old_fit_time == new_fit_time


def test_incompatibility(estimator_cfg: EstimatorConfig):
    estimator: Estimator = instantiate(estimator_cfg)
    assert False
