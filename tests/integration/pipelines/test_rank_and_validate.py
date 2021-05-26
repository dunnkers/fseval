import numpy as np
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


def test_rank_and_validate():
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
    cfg = OmegaConf.create(config.__dict__)

    callbacks: Callback = CallbackCollection()
    dataset: Dataset = Dataset(
        X=np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
        y=np.array([0, 1, 1, 0]),
        n=4,
        p=2,
        multioutput=False,
    )
    cv: CrossValidator = CrossValidator(
        name="train/test split",
        splitter=ShuffleSplit(n_splits=1, test_size=0.25, random_state=0),
    )
    storage_provider: AbstractStorageProvider = MockStorageProvider()

    pipeline = instantiate(cfg, callbacks, dataset, cv, storage_provider)

    X_train, X_test, y_train, y_test = cv.train_test_split(dataset.X, dataset.y)
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    assert list(score["best"].keys()) == ["validator"]
    assert score["best"]["validator"]["score"] == 1.0
