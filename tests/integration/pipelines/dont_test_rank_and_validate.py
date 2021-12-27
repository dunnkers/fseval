from typing import List, Tuple

import numpy as np
import pytest
from fseval.config import PipelineConfig
from fseval.pipeline.cv import CrossValidatorConfig
from fseval.pipeline.dataset import Dataset, DatasetConfig
from fseval.pipeline.estimator import EstimatorConfig, TaskedEstimatorConfig
from fseval.pipeline.resample import ResampleConfig
from fseval.types import (
    AbstractAdapter,
    AbstractEstimator,
    IncompatibilityError,
    Task,
)
from hydra.utils import instantiate
from omegaconf import OmegaConf


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

    def score(self, X, y):
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


@pytest.fixture
def classifier():
    estimator = dict(
        _target_="tests.integration.pipelines.test_rank_and_validate.MockRanker",
        random_state=0,
    )
    classifier: EstimatorConfig = EstimatorConfig(estimator=estimator)

    return classifier


@pytest.fixture
def ranker(classifier):
    ranker: TaskedEstimatorConfig = TaskedEstimatorConfig(
        name="Decision Tree",
        task=Task.classification,
        classifier=classifier,
        is_multioutput_dataset=False,
        estimates_feature_importances=True,
        estimates_feature_support=True,
        estimates_feature_ranking=True,
    )
    return ranker


@pytest.fixture
def validator(classifier):
    validator: TaskedEstimatorConfig = TaskedEstimatorConfig(
        name="Decision Tree",
        task=Task.classification,
        classifier=classifier,
        is_multioutput_dataset=False,
        estimates_target=True,
    )
    return validator


@pytest.fixture
def resample():
    resample: ResampleConfig = ResampleConfig(name="shuffle")
    return resample


@pytest.fixture
def dataset():
    return DatasetConfig(
        name="some_dataset",
        task=Task.classification,
        adapter={
            "_target_": "tests.integration.pipelines.test_rank_and_validate.MockAdapter"
        },
    )


@pytest.fixture
def cv():
    cv: CrossValidatorConfig = CrossValidatorConfig(
        name="train/test split",
        splitter=dict(
            _target_="sklearn.model_selection.ShuffleSplit",
            n_splits=1,
            test_size=0.25,
            random_state=0,
        ),
    )
    return cv


@pytest.fixture
def cfg(dataset, cv, resample, classifier, ranker, validator):
    config = PipelineConfig(
        pipeline="testing",
        dataset=dataset,
        cv=cv,
        callbacks=dict(
            _target_="fseval.pipelines._callback_collection.CallbackCollection"
        ),
        resample=resample,
        ranker=ranker,
        validator=validator,
        n_bootstraps=2,
        n_jobs=None,
        all_features_to_select="range(1, min(50, p) + 1)",
    )

    cfg = OmegaConf.create(config.__dict__)
    return cfg


def test_without_ranker_gt(cfg):
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


def test_with_ranker_gt(cfg):
    """Test execution with dataset ground-truth: a feature importances vector attached;
    i.e. the relevance per feature, known apriori."""
    cfg.dataset.feature_importances = {"X[:, :]": "1.0"}  # uniform

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


def test_with_ranker_gt_no_importances_substitution(cfg):
    """When no `feature_ranking` available, `feature_importances` should substitute
    for the ranking."""

    cfg.dataset.feature_importances = {"X[:, :]": "1.0"}  # uniform
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


def test_validator_incompatibility_check(cfg):
    with pytest.raises(IncompatibilityError):
        cfg.dataset.n = 5
        cfg.dataset.p = 5
        cfg.dataset.multioutput = False
        cfg.validator.estimates_target = False
        instantiate(cfg)


def test_ranker_incompatibility_check(cfg):
    with pytest.raises(IncompatibilityError):
        cfg.dataset.n = 5
        cfg.dataset.p = 5
        cfg.dataset.multioutput = False
        cfg.ranker.estimates_feature_importances = False
        cfg.ranker.estimates_feature_support = False
        cfg.ranker.estimates_feature_ranking = False
        instantiate(cfg)
