import numpy as np
import pytest
from fseval.config import ExperimentConfig
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.types import RunMode
from omegaconf import OmegaConf


@pytest.fixture(scope="session")
def cfg() -> ExperimentConfig:
    initialize(config_path="../fseval/conf")
    cfg: ExperimentConfig = compose(config_name="config")  # type: ignore
    return cfg


@pytest.fixture
def config_loader(cfg):
    gh = GlobalHydra.instance()  # is initialized by initialize/compose in `cfg`
    config_loader = gh.hydra.config_loader
    return config_loader


@pytest.fixture
def get_single_default(config_loader):
    """From the defaults fetched from all config sources, selects one by a given
    `config_path` name.
    """
    repo = config_loader.repository

    def _extract_default(config_path, config_name="config"):
        defaults_list = config_loader.compute_defaults_list(
            config_name, [], RunMode.RUN
        )
        defaults = defaults_list.defaults
        is_config_default = [default.config_path == config_path for default in defaults]
        config_default = np.extract(is_config_default, defaults)[0]
        default_config = config_loader._load_single_config(config_default, repo=repo)
        return default_config

    return _extract_default


@pytest.fixture
def structured_config(get_single_default):
    """Returns the structured config defined in `fseval/config.py`."""
    return get_single_default("base_config").config


@pytest.fixture
def get_single_config(config_loader, structured_config):
    """Returns the structured config defined in `fseval/config.py`."""

    def _get_single_config(group, path):
        cfg = OmegaConf.create()

        repo = config_loader.repository
        item_config = repo.load_config(f"{group}/{path}").config
        cfg[group] = item_config

        structured_config.merge_with(cfg)

        return structured_config[group]

    return _get_single_config
