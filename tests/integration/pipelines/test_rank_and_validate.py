import numpy as np
import pytest
from fseval.pipeline.cv import CrossValidator
from fseval.pipeline.dataset import Dataset
from fseval.pipeline.estimator import EstimatorConfig, TaskedEstimatorConfig
from fseval.pipeline.resample import ResampleConfig
from fseval.pipelines._callback_collection import CallbackCollection
from fseval.pipelines.rank_and_validate import RankAndValidateConfig
from fseval.storage_providers.mock import MockStorageProvider
from fseval.types import (
    AbstractEstimator,
    AbstractStorageProvider,
    Callback,
    IncompatibilityError,
    Task,
)
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.model_selection import ShuffleSplit


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
def pipeline_cfg(classifier, ranker, validator, resample):
    n_bootstraps: int = 2

    config = RankAndValidateConfig(
        resample=resample,
        ranker=ranker,
        validator=validator,
        n_bootstraps=n_bootstraps,
        all_features_to_select="range(1, min(50, p) + 1)",
    )

    pipeline_cfg = OmegaConf.create(config.__dict__)
    return pipeline_cfg


@pytest.fixture
def callbacks():
    callbacks: Callback = CallbackCollection()
    return callbacks


@pytest.fixture
def dataset_without_gt():
    """Dataset without ground-truth: a feature importances vector attached; i.e. the
    relevance per feature, known apriori."""
    dataset: Dataset = Dataset(
        name="some_dataset_name",
        X=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
        y=np.array([0, 1, 1, 0]),
        n=4,
        p=2,
        multioutput=False,
    )
    return dataset


@pytest.fixture
def dataset_with_gt(dataset_without_gt):
    """Dataset **with** ground-truth."""
    dataset_without_gt.feature_importances = np.array([0.5, 0.0, 0.5])
    return dataset_without_gt


@pytest.fixture
def cv():
    cv: CrossValidator = CrossValidator(
        name="train/test split",
        splitter=ShuffleSplit(n_splits=1, test_size=0.25, random_state=0),
    )
    return cv


@pytest.fixture
def storage_provider():
    storage_provider: AbstractStorageProvider = MockStorageProvider()
    return storage_provider

    return pipeline


def test_without_ranker_gt(
    pipeline_cfg, callbacks, dataset_without_gt, cv, storage_provider
):
    dataset = dataset_without_gt

    pipeline = instantiate(pipeline_cfg, callbacks, dataset, cv, storage_provider)
    X_train, X_test, y_train, y_test = cv.train_test_split(dataset.X, dataset.y)
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)

    assert score["best"]["validator"]["fit_time"] > 0
    assert score["best"]["validator"]["score"] >= 0.0


def test_with_ranker_gt(pipeline_cfg, callbacks, dataset_with_gt, cv, storage_provider):
    dataset = dataset_with_gt

    pipeline = instantiate(pipeline_cfg, callbacks, dataset, cv, storage_provider)
    X_train, X_test, y_train, y_test = cv.train_test_split(dataset.X, dataset.y)
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)

    assert score["best"]["validator"]["fit_time"] > 0
    assert score["best"]["ranker"]["fit_time"] > 0

    assert score["best"]["ranker"]["importance.r2_score"] <= 1.0
    assert score["best"]["ranker"]["importance.log_loss"] >= 0
    assert score["best"]["ranker"]["support.accuracy"] >= 0.0
    assert score["best"]["ranker"]["support.accuracy"] <= 1.0
    assert score["best"]["ranker"]["ranking.r2_score"] <= 1.0
    assert score["best"]["validator"]["score"] >= 0.0


def test_with_ranker_gt_no_importances_substitution(
    pipeline_cfg, callbacks, dataset_with_gt, cv, storage_provider
):
    """When no `feature_ranking` available, `feature_importances` should substitute
    for the ranking."""

    dataset = dataset_with_gt
    pipeline_cfg.ranker.estimates_feature_ranking = False

    pipeline = instantiate(pipeline_cfg, callbacks, dataset, cv, storage_provider)
    X_train, X_test, y_train, y_test = cv.train_test_split(dataset.X, dataset.y)
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)

    assert score["best"]["ranker"]["ranking.r2_score"] <= 1.0
    assert score["best"]["validator"]["score"] >= 0.0


def test_validator_incompatibility_check(
    pipeline_cfg, callbacks, dataset_with_gt, cv, storage_provider
):
    with pytest.raises(IncompatibilityError):
        pipeline_cfg["validator"]["estimates_target"] = False
        instantiate(pipeline_cfg, callbacks, dataset_with_gt, cv, storage_provider)


def test_ranker_incompatibility_check(
    pipeline_cfg, callbacks, dataset_with_gt, cv, storage_provider
):
    with pytest.raises(IncompatibilityError):
        pipeline_cfg["ranker"]["estimates_feature_importances"] = False
        pipeline_cfg["ranker"]["estimates_feature_support"] = False
        pipeline_cfg["ranker"]["estimates_feature_ranking"] = False
        instantiate(pipeline_cfg, callbacks, dataset_with_gt, cv, storage_provider)
