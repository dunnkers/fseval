import numpy as np
import pytest
from fseval.config import ExperimentConfig
from fseval.rankers import Ranker
from hydra import compose, initialize
from hydra.core.default_element import ResultDefault
from hydra.core.global_hydra import GlobalHydra
from hydra.types import RunMode
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


@pytest.fixture(scope="module", autouse=True)
def cfg() -> ExperimentConfig:
    initialize(config_path="../fseval/conf")
    cfg: ExperimentConfig = compose(config_name="config")  # type: ignore
    return cfg


@pytest.fixture(scope="module", autouse=True)
def config_loader(cfg):
    gh = GlobalHydra.instance()  # is initialized by initialize/compose in `cfg`
    config_loader = gh.hydra.config_loader
    return config_loader


def extract_default(config_loader, config_path, config_name="config"):
    repo = config_loader.repository
    defaults_list = config_loader.compute_defaults_list(config_name, [], RunMode.RUN)
    defaults = defaults_list.defaults
    is_config_default = [default.config_path == config_path for default in defaults]
    config_default = np.extract(is_config_default, defaults)[0]
    default_config = config_loader._load_single_config(config_default, repo=repo)
    return default_config


# how to parametrize pytest with group_options??


def get_base_config(config_loader):
    structured_config = extract_default(config_loader, "base_config")
    yaml_config = extract_default(config_loader, "config")
    return structured_config.merge_with(yaml_config)


@pytest.fixture(scope="module", autouse=True)
def group_options(config_loader, group):
    return config_loader.repository.get_group_options(group)


# @pytest.fixture(scope="module", autouse=True)
# def entire_config(config_loader):
#     base_config = get_base_config(config_loader)
#     group_options = group_options(config_loader, "ranker")

#     # groups = gh.hydra.list_all_config_groups()
#     # repo = config_loader.repository
#     # for group in groups:
#     #     print("group=", group)
#     #     for option in repo.get_group_options(group):
#     #         print("\toption=", option)
#     #         item = repo.load_config(f"{group}/{option}")
#     #         print("\t\titem=", item.config)
#     # pass

#     # base_config = get_base_config(config_loader, repo)
#     # cfg = OmegaConf.create()
#     # cfg.merge_with(loaded.config)
#     return


def test_all_rankers(group_options):
    pass


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
