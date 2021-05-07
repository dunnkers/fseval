from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from fseval.config import Task
from fseval.rankers import Ranker
from sklearn.base import BaseEstimator, clone, is_classifier, is_regressor


class SomeRankingEstimator(BaseEstimator):
    def fit(self, X, y):
        n, p = np.shape(X)
        self.feature_importances_ = np.ones(p)  # uniform ranking


@dataclass
class SomeRanker(Ranker):
    classifier: Any = SomeRankingEstimator()
    regressor: Any = SomeRankingEstimator()


@pytest.fixture
def ranker():
    return SomeRanker()


def test_initialization(ranker):
    ranker.task = Task.classification
    assert is_classifier(ranker)

    ranker.task = Task.regression
    assert is_regressor(ranker)

    config = ranker.get_config()
    assert "estimator" in config
    # should recursively get_params from other BaseEstimator or Configurable's
    assert isinstance(config["estimator"], dict)


def test_fit(ranker):
    ranker.task = Task.classification
    ranker.fit([[1, 1], [2, 1], [3, 3]], [0, 1, 1])
    assert sum(ranker.feature_importances_) >= 0


def test_can_clone(ranker):
    cloned_ranker = clone(ranker)
    assert cloned_ranker is not None
