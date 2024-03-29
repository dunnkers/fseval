import pytest
from omegaconf import DictConfig

from fseval.config import PipelineConfig
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
        ],
    )

    return config


def test_config_loading(cfg) -> None:
    assert isinstance(cfg, DictConfig)
    assert type(cfg) == DictConfig


def test_config_attributes(cfg) -> None:
    assert cfg.dataset is not None
    assert cfg.cv is not None
