from typing import Optional

import numpy as np
import pytest
from fseval.pipeline.estimator import Estimator, EstimatorConfig, TaskedEstimatorConfig
from fseval.types import Task
from fseval.utils.hydra_utils import TestGroupItem, generate_group_tests
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.base import clone


def pytest_generate_tests(metafunc):
    generate_group_tests("estimator", metafunc)


class TestEstimator(TestGroupItem):
    __test__ = False

    @staticmethod
    def _tasked_cfg(cfg, task: Task, is_multioutput_dataset: bool = False):
        tasked_config = TaskedEstimatorConfig(
            task=task, is_multioutput_dataset=is_multioutput_dataset
        )
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
    def estimator(self, cfg) -> Estimator:
        instance = instantiate(cfg)
        assert isinstance(instance, Estimator)
        return instance

    @pytest.fixture
    def X(self) -> np.ndarray:
        return np.array([[1, 2, 5], [-3, -4, 8], [5, 6, 1], [7, 8, 1], [1, 4, 5]])

    def test_clone(self, estimator):
        estimator_cloned = clone(estimator)
        assert isinstance(estimator_cloned, type(estimator))

    def test_fit(self, estimator: Estimator, X: np.ndarray, y: np.ndarray):
        estimator.fit(X, y)

        # estimator must either be some kind of feature ranker, or a predictor.
        assert (
            estimator.estimates_feature_importances
            or estimator.estimates_feature_support
            or estimator.estimates_feature_ranking
            or estimator.estimates_target
        )

        n, p = X.shape
        if estimator.estimates_feature_importances:
            feature_importances = estimator.feature_importances_
            assert np.shape(feature_importances) == (p,)

        if estimator.estimates_feature_support:
            feature_support = estimator.feature_support_
            assert np.shape(feature_support) == (p,)

        if estimator.estimates_feature_ranking:
            feature_ranking = estimator.feature_ranking_
            assert np.shape(feature_ranking) == (p,)

        if estimator.estimates_target:
            score = estimator.score(X, y)
            assert score > 0


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
    def y(self) -> np.ndarray:
        return np.array([0, 1, 0, 1, 0])


class TestMultioutputClassifiers(TestClassifiers):
    __test__ = True

    @staticmethod
    def get_cfg(cfg: DictConfig) -> Optional[DictConfig]:
        tasked_cfg = TestEstimator._tasked_cfg(
            cfg, Task.classification, is_multioutput_dataset=True
        )

        if tasked_cfg.classifier and (
            tasked_cfg.classifier.multioutput or tasked_cfg.multioutput
        ):
            return tasked_cfg
        else:
            return None

    @pytest.fixture
    def y(self) -> np.ndarray:
        return np.array([[0, 1], [1, 1], [0, 1], [1, 0], [1, 0]])


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
    def y(self) -> np.ndarray:
        return np.array([0.2, 1.5, 0.1, 1.7, 442.1])


class TestMultioutputRegressors(TestRegressors):
    __test__ = True

    @staticmethod
    def get_cfg(cfg: DictConfig) -> Optional[DictConfig]:
        tasked_cfg = TestEstimator._tasked_cfg(
            cfg, Task.regression, is_multioutput_dataset=True
        )

        if tasked_cfg.regressor and (
            tasked_cfg.regressor.multioutput or tasked_cfg.multioutput
        ):
            return tasked_cfg
        else:
            return None

    @pytest.fixture
    def y(self) -> np.ndarray:
        return np.array([[0.0, 1.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 3.0]])
