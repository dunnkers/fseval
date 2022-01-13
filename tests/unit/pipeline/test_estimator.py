import tempfile
from typing import cast

import pytest
from fseval.config import EstimatorConfig
from fseval.pipeline.estimator import Estimator
from fseval.storage.local import LocalStorage
from fseval.types import CacheUsage, IncompatibilityError, Task
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.base import BaseEstimator


@pytest.fixture
def estimator_cfg():
    estimator_config = EstimatorConfig(
        name="some_estimator",
        estimator={"_target_": "sklearn.tree.DecisionTreeClassifier"},
        _estimator_type="classifier",
        task=Task.classification,
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
    # classification estimator, but regression task
    estimator_cfg._estimator_type = "classifier"
    estimator_cfg.task = Task.regression
    with pytest.raises(IncompatibilityError):
        instantiate(estimator_cfg)

    # multioutput, but estimator does not support it (`multioutput=False`)
    estimator_cfg._estimator_type = "classifier"
    estimator_cfg.task = Task.classification
    estimator_cfg.multioutput = False
    estimator_cfg.is_multioutput_dataset = True
    with pytest.raises(IncompatibilityError):
        instantiate(estimator_cfg)

    # multioutput only, but
    estimator_cfg._estimator_type = "classifier"
    estimator_cfg.task = Task.classification
    estimator_cfg.multioutput_only = True
    estimator_cfg.is_multioutput_dataset = False
    with pytest.raises(IncompatibilityError):
        instantiate(estimator_cfg)


class FakeFeatureRanker(BaseEstimator):
    def fit(self, X, y):
        ...
        # does nothing! i.e. this fake ranker does not set the `feature_importances_`
        # or `coef_` attributes.


def test_invalid_feature_importances(estimator_cfg: EstimatorConfig):
    """When a ranker sets `estimates_feature_importances`, it should have one of two
    attributes after being fit: `feature_importances_` or `coef_`. If none are present
    on the estimator object, a `ValueError` should be thrown."""

    estimator_cfg.estimator = {
        "_target_": "tests.unit.pipeline.test_estimator.FakeFeatureRanker"
    }
    estimator_cfg.estimates_feature_importances = True

    # error should be raised when accessing `.feature_importances`
    estimator: Estimator = instantiate(estimator_cfg)
    with pytest.raises(ValueError):
        print(estimator.feature_importances_)  # trying to access `.feature_importances`
