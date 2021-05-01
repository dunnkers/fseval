from hydra.experimental import initialize, compose
from fseval.experiment import Experiment
from fseval.config import ExperimentConfig
from fseval.types import Ranker
from omegaconf import OmegaConf, DictConfig
import pytest


@pytest.fixture(scope="module", autouse=True)
def cfg() -> ExperimentConfig:
    initialize(config_path="../conf")
    cfg: ExperimentConfig = compose(config_name="config")  # type: ignore
    return cfg


def test_config_loading(cfg) -> None:
    assert isinstance(cfg, DictConfig)
    assert type(cfg) == DictConfig


def test_config_attributes(cfg) -> None:
    assert cfg.project is not None
    assert cfg.dataset is not None
    assert cfg.cv is not None
    assert cfg.bootstrap is not None
    assert cfg.ranker is not None


def test_instantiate_experiment(cfg) -> None:
    experiment = Experiment(cfg)
    assert experiment is not None
    assert isinstance(experiment.ranker, Ranker)
