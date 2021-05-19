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


class TestClassifiers(TestGroupItem):
    @staticmethod
    def get_cfg(cfg: DictConfig) -> Optional[DictConfig]:
        tasked_config = TaskedEstimatorConfig(task=Task.classification)
        tasked_cfg = OmegaConf.create(tasked_config.__dict__)
        tasked_cfg.merge_with(cfg)
        return tasked_cfg if tasked_cfg.classifier else None

    @pytest.fixture
    def estimator(self, cfg):
        estimator = instantiate(cfg)
        assert isinstance(estimator, Estimator)
        return estimator

    def test_fit(self, estimator):
        estimator.fit(np.array([[1, 2], [3, 4]]), np.array([0, 1]))


class TestRegressors(TestGroupItem):
    @staticmethod
    def get_cfg(cfg: DictConfig) -> Optional[DictConfig]:
        tasked_config = TaskedEstimatorConfig(task=Task.regression)
        tasked_cfg = OmegaConf.create(tasked_config.__dict__)
        tasked_cfg.merge_with(cfg)
        return tasked_cfg if tasked_cfg.regressor else None

    @pytest.fixture
    def estimator(self, cfg):
        estimator = instantiate(cfg)
        assert isinstance(estimator, Estimator)
        return estimator

    def test_fit(self, estimator):
        estimator.fit(np.array([[1, 2], [3, 4]]), np.array([0.2, 0.8]))
