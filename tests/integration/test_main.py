import os
import tempfile

import pytest
from fseval.config import EstimatorConfig, PipelineConfig
from fseval.main import run_pipeline
from fseval.types import IncompatibilityError
from fseval.utils.hydra_utils import get_config
from hydra.conf import ConfigStore


@pytest.fixture
def cfg() -> PipelineConfig:
    config = get_config(
        config_module="tests.integration.conf",
        config_name="empty_config",
        overrides=[
            "dataset=iris",
            "ranker=chi2",
            "validator=knn",
            "storage=mock",
        ],
    )

    return config


def test_run_pipeline(cfg: PipelineConfig):
    # execute from temporary dir
    tmpdir = tempfile.mkdtemp()
    os.chdir(tmpdir)

    # run pipeline
    run_pipeline(cfg)


def test_run_pipeline_n_jobs(cfg: PipelineConfig):
    """Run pipeline on multiple cores."""
    cfg.n_jobs = -1

    # execute from temporary dir
    tmpdir = tempfile.mkdtemp()
    os.chdir(tmpdir)

    # run pipeline
    run_pipeline(cfg)


@pytest.fixture
def incompatible_cfg() -> PipelineConfig:
    config = get_config(
        config_module="tests.integration.conf",
        config_name="empty_config",
        overrides=[
            "dataset=boston",
            "ranker=chi2",
            "validator=knn",
            "storage=mock",
        ],
    )

    return config


def test_pipeline_incompatibility(incompatible_cfg: PipelineConfig):
    """Pipeline should throw IncompatibilityError when trying to run a classification
    method on a regression dataset."""

    with pytest.raises(IncompatibilityError):
        run_pipeline(incompatible_cfg, raise_incompatibility_errors=True)


class SomeError(Exception):
    ...


class FailingRanker:
    def fit(self, X, y):
        raise SomeError(
            "hi! im a feature ranker. but im not having a good day, so im just going "
            + "to raise an error like this. cheers! 🍻"
        )


@pytest.fixture
def failing_cfg() -> PipelineConfig:
    # failing ranker
    failing_ranker = EstimatorConfig(
        name="FailingRanker",
        estimator={
            "_target_": "tests.integration.test_main.FailingRanker",
        },
        _estimator_type="classifier",
        estimates_feature_importances=True,
    )

    # store in config store
    cs = ConfigStore.instance()
    cs.store(name="failing_ranker", node=failing_ranker, group="ranker")

    # construct pipeline config
    config = get_config(
        config_module="tests.integration.conf",
        config_name="empty_config",
        overrides=[
            "dataset=iris",
            "ranker=failing_ranker",
            "validator=knn",
            "storage=mock",
        ],
    )

    return config


def test_pipeline_failure(failing_cfg: PipelineConfig):
    with pytest.raises(SomeError):
        run_pipeline(failing_cfg)
