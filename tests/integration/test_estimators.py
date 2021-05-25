from typing import Optional

import numpy as np
import pytest
from fseval.pipeline.estimator import Estimator, EstimatorConfig, TaskedEstimatorConfig
from fseval.types import Task
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import minmax_scale
from tests.integration.hydra_utils import TestGroupItem, generate_group_tests


def pytest_generate_tests(metafunc):
    generate_group_tests("estimator", metafunc)


class TestEstimator(TestGroupItem):
    __test__ = False

    @staticmethod
    def _tasked_cfg(cfg, task):
        tasked_config = TaskedEstimatorConfig(task=task)
        tasked_cfg = OmegaConf.create(tasked_config.__dict__)
        tasked_cfg.merge_with(cfg)

        if tasked_cfg.classifier:
            estimator_config = EstimatorConfig()
            estimator_cfg = OmegaConf.create(estimator_config.__dict__)
            estimator_cfg.merge_with(tasked_cfg.classifier)
            tasked_cfg.classifier = estimator_cfg

        if tasked_cfg.regressor:
            estimator_config = EstimatorConfig()
            estimator_cfg = OmegaConf.create(estimator_config.__dict__)
            estimator_cfg.merge_with(tasked_cfg.regressor)
            tasked_cfg.regressor = estimator_cfg

        return tasked_cfg

    @pytest.fixture
    def estimator(self, cfg):
        instance = instantiate(cfg)
        assert isinstance(instance, Estimator)
        return instance

    @pytest.fixture
    def X(self):
        return np.array([[1, 2], [-3, -4], [5, 6], [7, 8]])

    def test_fit(self, estimator, X, y):
        if estimator._get_tags().get("requires_positive_X"):
            X = minmax_scale(X)

        estimator.fit(X, y)


class TestClassifiers(TestEstimator):
    __test__ = True

    @staticmethod
    def get_cfg(cfg: DictConfig) -> Optional[DictConfig]:
        tasked_cfg = TestEstimator._tasked_cfg(cfg, Task.classification)

        if tasked_cfg.classifier and not tasked_cfg.classifier.multioutput_only:
            return tasked_cfg
        else:
            return None

    @pytest.fixture
    def y(self):
        return np.array([0, 1, 0, 1])


class TestMultioutputClassifiers(TestClassifiers):
    __test__ = True

    @staticmethod
    def get_cfg(cfg: DictConfig) -> Optional[DictConfig]:
        tasked_cfg = TestEstimator._tasked_cfg(cfg, Task.classification)

        if tasked_cfg.classifier and tasked_cfg.classifier.multioutput:
            return tasked_cfg
        else:
            return None

    @pytest.fixture
    def y(self):
        return np.array([[0, 1], [1, 1], [0, 1], [1, 0]])


class TestRegressors(TestEstimator):
    __test__ = True

    @staticmethod
    def get_cfg(cfg: DictConfig) -> Optional[DictConfig]:
        tasked_cfg = TestEstimator._tasked_cfg(cfg, Task.regression)

        if tasked_cfg.regressor and not tasked_cfg.regressor.multioutput_only:
            return tasked_cfg
        else:
            return None

    @pytest.fixture
    def y(self):
        return np.array([0.0, 1.0, 0.0, 1.0])


class TestMultioutputRegressors(TestRegressors):
    __test__ = True

    @staticmethod
    def get_cfg(cfg: DictConfig) -> Optional[DictConfig]:
        tasked_cfg = TestEstimator._tasked_cfg(cfg, Task.regression)

        if tasked_cfg.regressor and tasked_cfg.regressor.multioutput:
            return tasked_cfg
        else:
            return None

    @pytest.fixture
    def y(self):
        return np.array([[0.0, 1.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
