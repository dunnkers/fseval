from typing import List, cast

import numpy as np
import pytest
from fseval.config import RankerConfig, Task
from fseval.rankers import Ranker
from hydra._internal.hydra import Hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from tests.utils import get_group_configs

# ALL_RANKERS = [
#     dict(name="chi2", classifier=True, regressor=False),
#     dict(name="relieff", classifier=True, regressor=True),
#     dict(name="tabnet", classifier=True, regressor=True),
# ]
# ALL_CLASSIFIERS =
ALL_RANKERS = np.array(
    [("chi2", True, False), ("relieff", True, True), ("tabnet", True, True)],
    dtype=[("name", str), ("classifier", bool), ("regressor", bool)],
)
ALL_RANKER_NAMES = ALL_RANKERS["name"]


def pytest_generate_tests(metafunc):
    all_ranker_cfgs = get_group_configs("ranker")
    all_ranker_cfgs = cast(List[RankerConfig], all_ranker_cfgs)

    pytest_ids = []
    argvalues = []

    for ranker_cfg in all_ranker_cfgs:
        ranker_cfg = metafunc.cls.get_ranker_cfg(ranker_cfg)
        if ranker_cfg:
            pytest_ids.append(ranker_cfg.name)
            argvalues.append([ranker_cfg])

    metafunc.parametrize(["ranker_cfg"], argvalues, ids=pytest_ids, scope="class")


class TestScenario:
    @staticmethod
    def get_ranker_cfg(ranker_cfg: RankerConfig) -> RankerConfig:
        return ranker_cfg


class TestClassifiers(TestScenario):
    @staticmethod
    def get_ranker_cfg(ranker_cfg: RankerConfig):
        if ranker_cfg.classifier is not None:
            ranker_cfg.task = Task.classification
            return ranker_cfg

    def test_initialization(self, ranker_cfg):
        ranker = instantiate(ranker_cfg)
        assert isinstance(ranker, Ranker)

    # def test_demo1(self, attribute):
    #     assert isinstance(attribute, str)

    # def test_demo2(self, attribute):
    #     assert isinstance(attribute, str)


# @pytest.fixture(params=ALL_RANKER_NAMES)
# def ranker_cfg(config_repo, request):
#     single_config = get_single_config(config_repo, "ranker", request.param)
#     return single_config


# def test_initialization(ranker_cfg):
#     ranker_cfg["task"] = Task.classification
#     ranker = instantiate(ranker_cfg)
#     assert isinstance(ranker, Ranker)

#     ranker_cfg["task"] = Task.regression
#     ranker = instantiate(ranker_cfg)
#     assert isinstance(ranker, Ranker)


# @pytest.fixture
# def clf_ranker(ranker_cfg):
#     ranker_cfg["task"] = Task.classification
#     return instantiate(ranker_cfg)


# @pytest.fixture
# def reg_ranker(ranker_cfg):
#     ranker_cfg["task"] = Task.regression
#     return instantiate(ranker_cfg)


# @pytest.fixture
# def X():
#     return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


# @pytest.fixture
# def clf_y():
#     return np.array([0, 1, 1])


# @pytest.fixture
# def reg_y():
#     return np.array([0.0, 1.0, 1.0])


# def test_classifier_fit(clf_ranker, X, clf_y):
#     clf_ranker.fit(X, clf_y)


# def test_regressor_fit(reg_ranker, X, reg_y):
#     reg_ranker.fit(X, reg_y)


# def test_feature_importances(clf_ranker, reg_ranker, X, clf_y, reg_y):
#     clf_ranker.fit(X, clf_y)
#     assert np.isclose(sum(clf_ranker.feature_importances_), 1)

#     reg_ranker.fit(X, reg_y)
#     assert np.isclose(sum(reg_ranker.feature_importances_), 1)
