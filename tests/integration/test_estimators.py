from typing import Optional

import numpy as np
import pytest
from fseval.pipeline.estimator import Estimator, EstimatorConfig, TaskedEstimatorConfig
from fseval.types import Task
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tests.integration.hydra_utils import TestGroupItem, generate_group_tests


def pytest_generate_tests(metafunc):
    generate_group_tests("estimator", metafunc)


class TestEstimator(TestGroupItem):
    __test__ = False

    @pytest.fixture
    def estimator(self, cfg):
        instance = instantiate(cfg)
        assert isinstance(instance, Estimator)
        return instance


class TestClassifiers(TestEstimator):
    __test__ = True

    @staticmethod
    def get_cfg(cfg: DictConfig) -> Optional[DictConfig]:
        tasked_config = TaskedEstimatorConfig(task=Task.classification)
        tasked_cfg = OmegaConf.create(tasked_config.__dict__)
        tasked_cfg.merge_with(cfg)
        return tasked_cfg if tasked_cfg.classifier else None

    def test_fit(self, estimator):
        estimator.fit(np.array([[1, 2], [3, 4]]), np.array([0, 1]))


class TestRegressors(TestEstimator):
    __test__ = True

    @staticmethod
    def get_cfg(cfg: DictConfig) -> Optional[DictConfig]:
        tasked_config = TaskedEstimatorConfig(task=Task.regression)
        tasked_cfg = OmegaConf.create(tasked_config.__dict__)
        tasked_cfg.merge_with(cfg)
        return tasked_cfg if tasked_cfg.regressor else None

    def test_fit(self, estimator):
        estimator.fit(np.array([[1, 2], [3, 4]]), np.array([0.2, 0.8]))
