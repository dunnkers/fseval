from typing import cast

import numpy as np
import pytest
from fseval.config import ExperimentConfig
from fseval.experiment import Experiment
from fseval.rankers import Ranker
from hydra.utils import instantiate
from omegaconf import DictConfig

from tests.hydra_utils import get_config


@pytest.fixture
def cfg() -> DictConfig:
    return get_config()


def test_config_loading(cfg) -> None:
    assert isinstance(cfg, DictConfig)
    assert type(cfg) == DictConfig


def test_config_attributes(cfg) -> None:
    cfg = cast(ExperimentConfig, cfg)
    assert cfg.project is not None
    assert cfg.dataset is not None
    assert cfg.cv is not None
    assert cfg.resample is not None
    assert cfg.ranker is not None
    assert cfg.validator is not None


def test_instantiate_experiment(cfg) -> None:
    experiment = instantiate(cfg)
    assert experiment is not None
    assert isinstance(experiment, Experiment)
    assert isinstance(experiment.ranker, Ranker)


def test_experiment_params(cfg) -> None:
    experiment = instantiate(cfg)
    params = experiment.get_params()
    assert params["dataset"] is not None
    assert params["cv__fold"] == 0


def test_experiment_config(cfg) -> None:
    experiment = instantiate(cfg)
    config = experiment.get_config()
    assert config["dataset"] is not None
    assert isinstance(config["ranker"]["estimator"], dict)
    assert isinstance(config["dataset"]["adapter"], dict)
    # None should be removed, just like MISSING
    assert not hasattr(config["validator"], "min_impurity_split")
