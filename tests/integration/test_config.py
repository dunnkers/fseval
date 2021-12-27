import pytest
from fseval.config import PipelineConfig
from fseval.utils.hydra_utils import get_config, get_single_config
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig


@pytest.fixture
def cfg() -> PipelineConfig:
    cs = ConfigStore.instance()
    cs.store(name="my_config", node=PipelineConfig())
    config = get_config(
        config_name="my_config",
        overrides=[
            "+dataset=iris",
            "+estimator@ranker=chi2",
            "+estimator@validator=decision_tree",
        ],
    )

    return config


def test_config_loading(cfg) -> None:
    assert isinstance(cfg, DictConfig)
    assert type(cfg) == DictConfig


def test_config_attributes(cfg) -> None:
    assert cfg.dataset is not None
    assert cfg.cv is not None
    assert cfg.pipeline is not None
