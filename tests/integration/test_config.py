import numpy as np
import pytest
from hydra.utils import instantiate
from omegaconf import DictConfig

from fseval.config import BaseConfig
from tests.integration.hydra_utils import get_config


@pytest.fixture
def cfg() -> BaseConfig:
    return get_config()


def test_config_loading(cfg) -> None:
    assert isinstance(cfg, DictConfig)
    assert type(cfg) == DictConfig


def test_config_attributes(cfg) -> None:
    assert cfg.dataset is not None
    assert cfg.cv is not None
    assert cfg.pipeline is not None
