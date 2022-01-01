import os
import tempfile

import pytest
from fseval.config import PipelineConfig
from fseval.main import run_pipeline
from fseval.types import IncompatibilityError
from fseval.utils.hydra_utils import get_config


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
