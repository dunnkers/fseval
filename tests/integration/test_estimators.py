import itertools

import numpy as np
import pytest
from fseval.config import EstimatorConfig, PipelineConfig
from fseval.pipeline.estimator import Estimator
from fseval.types import Task
from fseval.utils.hydra_utils import get_group_pipeline_configs
from hydra.utils import instantiate
from sklearn.base import clone

from ._group_test_utils import ShouldTestGroupItem


def pytest_generate_tests(metafunc):
    ## Add all rankers
    ranker_argvalues, ranker_pytest_ids = get_group_pipeline_configs(
        config_module="tests.integration.conf",
        config_name="simple_defaults",
        group_name="ranker",
        should_test=metafunc.cls.should_test,
    )
    ranker_argvalues = list(zip(ranker_argvalues, itertools.repeat("ranker")))

    ## Add all validators
    validator_argvalues, validator_pytest_ids = get_group_pipeline_configs(
        config_module="tests.integration.conf",
        config_name="simple_defaults",
        group_name="validator",
        should_test=metafunc.cls.should_test,
    )
    validator_argvalues = list(zip(validator_argvalues, itertools.repeat("validator")))

    ## Merge rankers and validators
    pytest_ids = ranker_pytest_ids + validator_pytest_ids
    pytest_ids, unique_ids = np.unique(pytest_ids, return_index=True)
    argvalues = ranker_argvalues + validator_argvalues
    argvalues = [argvalues[i] for i in unique_ids]

    ## Parametrize using pytest metafunc
    metafunc.parametrize(
        ["cfg", "group_name"],
        argvalues,
        ids=pytest_ids,
        scope="class",
    )


class TestEstimator(ShouldTestGroupItem):
    __test__ = False

    @pytest.fixture
    def estimator(self, cfg: PipelineConfig, group_name: str) -> Estimator:
        """Retrieve the relevant config and instantiate pipeline."""

        # set dataset properties to some random number. instantiating the pipeline
        # requires these numbers to be set. emulates having loaded the dataset.
        cfg.dataset.n = 999
        cfg.dataset.p = 999
        cfg.dataset.multioutput = False
        pipeline = instantiate(cfg)

        # instantiate estimator
        estimator = getattr(pipeline, group_name)
        assert isinstance(estimator, Estimator)

        return estimator

    @pytest.fixture
    def X(self) -> np.ndarray:
        return np.array([[1, 2, 5], [-3, -4, 8], [5, 6, 1], [7, 8, 1], [1, 4, 5]])

    def test_clone(self, estimator: Estimator):
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

            assert isinstance(score, float)
            assert score > 0.0


class TestClassifiers(TestEstimator):
    __test__ = True

    @staticmethod
    def should_test(cfg: PipelineConfig, group_name: str) -> bool:
        estimator_cfg: EstimatorConfig = getattr(cfg, group_name)
        _estimator_type = estimator_cfg._estimator_type
        cfg.dataset.task = Task.classification

        return _estimator_type == "classifier" and not estimator_cfg.multioutput_only

    @pytest.fixture
    def y(self) -> np.ndarray:
        return np.array([0, 1, 0, 1, 0])


class TestMultioutputClassifiers(TestClassifiers):
    __test__ = True

    @staticmethod
    def should_test(cfg: PipelineConfig, group_name: str) -> bool:
        estimator_cfg: EstimatorConfig = getattr(cfg, group_name)
        _estimator_type = estimator_cfg._estimator_type
        cfg.dataset.task = Task.classification

        return _estimator_type == "classifier" and estimator_cfg.multioutput

    @pytest.fixture
    def y(self) -> np.ndarray:
        return np.array([[0, 1], [1, 1], [0, 1], [1, 0], [1, 0]])


class TestRegressors(TestEstimator):
    __test__ = True

    @staticmethod
    def should_test(cfg: PipelineConfig, group_name: str) -> bool:
        estimator_cfg: EstimatorConfig = getattr(cfg, group_name)
        _estimator_type = estimator_cfg._estimator_type
        cfg.dataset.task = Task.regression

        return _estimator_type == "regressor" and not estimator_cfg.multioutput_only

    @pytest.fixture
    def y(self) -> np.ndarray:
        return np.array([0.2, 1.5, 0.1, 1.7, 442.1])


class TestMultioutputRegressors(TestRegressors):
    __test__ = True

    @staticmethod
    def should_test(cfg: PipelineConfig, group_name: str) -> bool:
        estimator_cfg: EstimatorConfig = getattr(cfg, group_name)
        _estimator_type = estimator_cfg._estimator_type
        cfg.dataset.task = Task.regression

        return _estimator_type == "regressor" and estimator_cfg.multioutput

    @pytest.fixture
    def y(self) -> np.ndarray:
        return np.array([[0.0, 1.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 3.0]])
