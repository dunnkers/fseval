from typing import List, cast

import numpy as np
import pytest
from fseval.config import GroupItem, RankerConfig, Task
from fseval.rankers import Ranker
from hydra._internal.hydra import Hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from tests.integration.hydra_utils import TestGroupItem, generate_group_tests


def pytest_generate_tests(metafunc):
    generate_group_tests("ranker", metafunc)


class TestRanker(TestGroupItem):
    __test__ = False

    def test_initialization(self, cfg):
        ranker = instantiate(cfg)
        assert isinstance(ranker, Ranker)

    @pytest.fixture
    def ranker(self, cfg):
        ranker = instantiate(cfg)
        return ranker

    def test_fit(self, ranker, X, y):
        ranker.fit(X, y)
        summed = sum(ranker.feature_importances_)
        assert np.isclose(summed, 1)


@pytest.fixture
def X():
    return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


class TestClassifiers(TestRanker):
    __test__ = True

    @staticmethod
    def get_cfg(cfg: DictConfig):
        if cfg.classifier is not None:
            cfg.task = Task.classification
            return cfg

    @pytest.fixture
    def y(self):
        return np.array([0, 1, 1])


class TestRegressors(TestRanker):
    __test__ = True

    @staticmethod
    def get_cfg(cfg: DictConfig):
        if cfg.regressor is not None:
            cfg.task = Task.regression
            return cfg

    @pytest.fixture
    def y(self):
        return np.array([0.0, 1.0, 1.0])
