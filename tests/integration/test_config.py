import pytest
from fseval.config import BaseConfig
from fseval.utils.hydra_utils import get_config, get_single_config
from omegaconf import DictConfig


@pytest.fixture
def cfg() -> BaseConfig:
    config = get_config(
        overrides=[
            "+dataset=iris",
            "+estimator@ranker=chi2",
            "+estimator@validator=decision_tree",
        ]
    )

    return config


def test_config_loading(cfg) -> None:
    assert isinstance(cfg, DictConfig)
    assert type(cfg) == DictConfig


def test_config_attributes(cfg) -> None:
    assert cfg.dataset is not None
    assert cfg.cv is not None
    assert cfg.pipeline is not None
