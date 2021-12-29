from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import pytest
from fseval.config import (
    CrossValidatorConfig,
    DatasetConfig,
    EstimatorConfig,
    PipelineConfig,
    ResampleConfig,
    TaskedEstimatorConfig,
)
from fseval.pipeline.dataset import Dataset, DatasetLoader
from fseval.types import AbstractAdapter, AbstractEstimator, IncompatibilityError, Task
from fseval.utils.hydra_utils import get_config
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf

cs = ConfigStore.instance()


class MockRanker(AbstractEstimator):
    def __init__(self, random_state=None):
        self.random_state = random_state

    def _get_random_state(self):
        return np.random.RandomState(self.random_state)

    def fit(self, X, y):
        n, p = np.asarray(X).shape
        self.n_features = p

    def transform(self, X, y):
        ...

    def fit_transform(self, X, y):
        ...

    def score(self, X, y, **kwargs) -> Union[Dict, pd.DataFrame, np.generic, None]:
        return self._get_random_state().rand()

    @property
    def feature_importances_(self):
        return self._get_random_state().rand(self.n_features)

    @property
    def support_(self):
        return self._get_random_state().rand(self.n_features)

    @property
    def ranking_(self):
        return self._get_random_state().rand(self.n_features)


class MockAdapter(AbstractAdapter):
    def get_data(self) -> Tuple[List, List]:
        X = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        y = [0, 1, 1, 0]
        return X, y


estimator = dict(
    _target_="tests.integration.pipelines.test_rank_and_validate.MockRanker",
    random_state=0,
)
classifier: EstimatorConfig = EstimatorConfig(estimator=estimator)
ranker: TaskedEstimatorConfig = TaskedEstimatorConfig(
    name="Decision Tree",
    task=Task.classification,
    classifier=classifier,
    is_multioutput_dataset=False,
    estimates_feature_importances=True,
    estimates_feature_support=True,
    estimates_feature_ranking=True,
)
cs.store(name="decision_tree", node=ranker, group="ranker")

validator: TaskedEstimatorConfig = TaskedEstimatorConfig(
    name="Decision Tree",
    task=Task.classification,
    classifier=classifier,
    is_multioutput_dataset=False,
    estimates_target=True,
)
cs.store(name="decision_tree", node=validator, group="validator")

resample: ResampleConfig = ResampleConfig(name="shuffle")
cs.store(name="shuffle", node=resample, group="resample")

dataset: DatasetConfig = DatasetConfig(
    name="some_dataset",
    task=Task.classification,
    adapter={
        "_target_": "tests.integration.pipelines.test_rank_and_validate.MockAdapter"
    },
)
cs.store(name="some_dataset", node=dataset, group="dataset")

cv: CrossValidatorConfig = CrossValidatorConfig(
    name="train/test split",
    splitter=dict(
        _target_="sklearn.model_selection.ShuffleSplit",
        n_splits=1,
        test_size=0.25,
        random_state=0,
    ),
)
cs.store(name="train_test_split", node=cv, group="cv")

config = PipelineConfig(
    pipeline="testing",
    n_bootstraps=2,
    n_jobs=None,
    all_features_to_select="range(1, min(50, p) + 1)",
)
cs.store(name="my_test_config", node=config)


@pytest.fixture
def cfg() -> PipelineConfig:
    cfg: PipelineConfig = get_config(
        config_module="tests.integration.pipelines.conf",
        config_name="my_test_config",
        overrides=[
            "dataset=some_dataset",
            "cv=train_test_split",
            "validator=decision_tree",
            "ranker=decision_tree",
            "resample=shuffle",
        ],
    )

    return cfg


def test_without_ranker_gt(cfg: PipelineConfig):
    """Test execution without dataset ground-truth."""

    # load dataset
    dataset_loader: DatasetLoader = instantiate(cfg.dataset)
    dataset: Dataset = dataset_loader.load()
    cfg.dataset.n = dataset.n
    cfg.dataset.p = dataset.p
    cfg.dataset.multioutput = dataset.multioutput

    # fit pipeline
    pipeline = instantiate(cfg)
    X_train, X_test, y_train, y_test = pipeline.cv.train_test_split(
        dataset.X, dataset.y
    )
    pipeline.fit(X_train, y_train)
    pipeline.score(X_test, y_test, feature_importances=dataset.feature_importances)


def test_with_ranker_gt(cfg: PipelineConfig):
    """Test execution with dataset ground-truth: a feature importances vector attached;
    i.e. the relevance per feature, known apriori."""
    cfg.dataset.feature_importances = {"X[:, :]": 1.0}  # uniform

    # load dataset
    dataset_loader: DatasetLoader = instantiate(cfg.dataset)
    dataset: Dataset = dataset_loader.load()
    cfg.dataset.n = dataset.n
    cfg.dataset.p = dataset.p
    cfg.dataset.multioutput = dataset.multioutput

    # fit pipeline
    pipeline = instantiate(cfg)
    X_train, X_test, y_train, y_test = pipeline.cv.train_test_split(
        dataset.X, dataset.y
    )
    pipeline.fit(X_train, y_train)
    pipeline.score(X_test, y_test, feature_importances=dataset.feature_importances)


def test_with_ranker_gt_no_importances_substitution(cfg: PipelineConfig):
    """When no `feature_ranking` available, `feature_importances` should substitute
    for the ranking."""

    cfg.dataset.feature_importances = {"X[:, :]": 1.0}  # uniform
    cfg.ranker.estimates_feature_ranking = False

    # load dataset
    dataset_loader: DatasetLoader = instantiate(cfg.dataset)
    dataset: Dataset = dataset_loader.load()
    cfg.dataset.n = dataset.n
    cfg.dataset.p = dataset.p
    cfg.dataset.multioutput = dataset.multioutput

    # fit pipeline
    pipeline = instantiate(cfg)
    X_train, X_test, y_train, y_test = pipeline.cv.train_test_split(
        dataset.X, dataset.y
    )
    pipeline.fit(X_train, y_train)
    pipeline.score(X_test, y_test, feature_importances=dataset.feature_importances)


def test_validator_incompatibility_check(cfg: PipelineConfig):
    with pytest.raises(IncompatibilityError):
        cfg.dataset.n = 5
        cfg.dataset.p = 5
        cfg.dataset.multioutput = False
        cfg.validator.estimates_target = False
        instantiate(cfg)


def test_ranker_incompatibility_check(cfg: PipelineConfig):
    with pytest.raises(IncompatibilityError):
        cfg.dataset.n = 5
        cfg.dataset.p = 5
        cfg.dataset.multioutput = False
        cfg.ranker.estimates_feature_importances = False
        cfg.ranker.estimates_feature_support = False
        cfg.ranker.estimates_feature_ranking = False
        instantiate(cfg)
