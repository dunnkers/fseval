from typing import cast

import pytest
from fseval.config import ExperimentConfig
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra


@pytest.fixture(scope="session")
def cfg() -> ExperimentConfig:
    initialize(config_path="../fseval/conf")
    config = compose(config_name="config")
    cfg: ExperimentConfig = cast(ExperimentConfig, config)
    return cfg


@pytest.fixture
def config_repo(cfg):
    gh = GlobalHydra.instance()  # is initialized by initialize/compose in `cfg`
    config_loader = gh.hydra.config_loader
    repo = config_loader.repository
    return repo
