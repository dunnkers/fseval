from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from fseval.config import ExperimentConfig
from fseval.rankers import Ranker
import pytest
from hydra import initialize, compose

from hydra.core.global_hydra import GlobalHydra
from hydra.core.default_element import ResultDefault
from hydra.types import RunMode
import numpy as np


@pytest.fixture(scope="module", autouse=True)
def cfg() -> ExperimentConfig:
    initialize(config_path="../fseval/conf")
    cfg: ExperimentConfig = compose(config_name="config")  # type: ignore
    return cfg


def test_single_config_loading(cfg):
    gh = GlobalHydra.instance()
    groups = gh.hydra.list_all_config_groups()
    config_loader = gh.hydra.config_loader
    repo = config_loader.repository
    for group in groups:
        print("group=", group)
        for option in repo.get_group_options(group):
            print("\toption=", option)
            item = repo.load_config(f"{group}/{option}")
            print("\t\titem=", item.config)
    pass

    # merge with:
    defaults_list = config_loader.compute_defaults_list("config", [], RunMode.RUN)
    defaults = defaults_list.defaults
    is_config_default = [default.config_path == "config" for default in defaults]
    config_default = np.extract(is_config_default, defaults)[0]
    default_config = config_loader._load_single_config(config_default, repo=repo)
    print("defaults to merge=", default_config.config)
    # cfg = OmegaConf.create()
    # cfg.merge_with(loaded.config)
    pass


# get defaults config_loader.compute_defaults_list('config', [], run_mode=RunMode.RUN)


def test_config_loading(cfg) -> None:
    assert isinstance(cfg, DictConfig)
    assert type(cfg) == DictConfig


def test_config_attributes(cfg) -> None:
    assert cfg.project is not None
    assert cfg.dataset is not None
    assert cfg.cv is not None
    assert cfg.resample is not None
    assert cfg.ranker is not None
    assert cfg.validator is not None


def test_instantiate_experiment(cfg) -> None:
    experiment = instantiate(cfg)
    assert experiment is not None
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
