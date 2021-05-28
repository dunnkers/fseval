import numpy as np
import pytest
from fseval.pipeline.cv import CrossValidator
from fseval.pipeline.dataset import Dataset
from fseval.pipeline.estimator import EstimatorConfig, TaskedEstimatorConfig
from fseval.pipeline.resample import ResampleConfig
from fseval.pipelines._callback_collection import CallbackCollection
from fseval.pipelines.rank_and_validate import RankAndValidateConfig
from fseval.storage_providers.mock import MockStorageProvider
from fseval.types import AbstractStorageProvider, Callback, Task
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.model_selection import ShuffleSplit


@pytest.fixture
def pipeline_cfg():
    estimator = dict(_target_="sklearn.tree.DecisionTreeClassifier", random_state=0)
    classifier: EstimatorConfig = EstimatorConfig(estimator=estimator)

    resample: ResampleConfig = ResampleConfig(name="shuffle")
    ranker: TaskedEstimatorConfig = TaskedEstimatorConfig(
        name="dt",
        task=Task.classification,
        classifier=classifier,
        is_multioutput_dataset=False,
    )
    validator: TaskedEstimatorConfig = TaskedEstimatorConfig(
        name="dt",
        task=Task.classification,
        classifier=classifier,
        is_multioutput_dataset=False,
    )
    n_bootstraps: int = 2

    config = RankAndValidateConfig(
        resample=resample, ranker=ranker, validator=validator, n_bootstraps=n_bootstraps
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
    assert score["best"]["validator"]["score"] == 1.0


def test_with_ranker_gt(pipeline_cfg, callbacks, dataset_with_gt, cv, storage_provider):
    dataset = dataset_with_gt

    pipeline = instantiate(pipeline_cfg, callbacks, dataset, cv, storage_provider)
    X_train, X_test, y_train, y_test = cv.train_test_split(dataset.X, dataset.y)
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)

    assert score["best"]["validator"]["fit_time"] > 0
    assert score["best"]["ranker"]["fit_time"] > 0

    assert score["best"]["ranker"]["r2_score"] <= 1.0
    assert score["best"]["validator"]["score"] == 1.0
