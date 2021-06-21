from typing import List, Optional

from fseval.config import BaseConfig
from hydra import compose, initialize_config_module
from hydra.core.config_loader import ConfigLoader
from hydra.core.global_hydra import GlobalHydra
from hydra.core.object_type import ObjectType
from hydra.types import RunMode
from omegaconf import DictConfig, OmegaConf

"""Helper functions for Hydra-related operations. e.g. finding out what options are
available for some group, or getting a specific config for some group item. Enables
fully automated testing of group configs.

Author: Jeroen Overschie"""


def _ensure_hydra_initialized() -> None:
    """Initializes Hydra only if it is not already initialized."""
    gh = GlobalHydra()
    if not gh.is_initialized():
        initialize_config_module(config_module="fseval.conf", job_name="fseval_tests")


def _get_config_loader() -> ConfigLoader:
    """Grabs the Hydra `ConfigLoader`."""
    _ensure_hydra_initialized()
    gh = GlobalHydra()
    cl = gh.config_loader()
    return cl


def get_config(
    config_name: Optional[str] = "my_config", overrides: List[str] = []
) -> BaseConfig:
    """Gets the fseval configuration as composed by Hydra. Local .yaml configuration
    and defaults are automatically merged."""
    _ensure_hydra_initialized()
    config = compose(config_name=config_name, overrides=overrides)
    return config  # type: ignore


def get_single_config(group_name: str, option_name: str) -> DictConfig:
    """Returns a single config, e.g. `ranker/chi2` with the structured config
    defined in `fseval/config.py` merged into it."""
    cl = _get_config_loader()
    cfg = OmegaConf.create()

    structured_config = cl.load_configuration("base_config", [], RunMode.RUN)
    OmegaConf.set_struct(structured_config, False)
    cfg.merge_with(structured_config)

    item_config = cl.load_configuration(f"{group_name}/{option_name}", [], RunMode.RUN)
    cfg.merge_with(item_config)

    return cfg[group_name]


def get_group_options(
    group_name: str, results_filter: Optional[ObjectType] = ObjectType.CONFIG
) -> List[str]:
    """Gets the options for a certain grouop.

    e.g. `get_group_options(<dataset_name>)` returns a list with all
    available dataset names for use in fseval."""
    cl = _get_config_loader()
    group_options = cl.get_group_options(group_name)
    return group_options


def get_group_configs(
    group_name: str, results_filter: Optional[ObjectType] = ObjectType.CONFIG
) -> List[DictConfig]:
    """Gets the configs related to a certain group, as `DictConfig`'s."""
    group_options = get_group_options(group_name)
    group_configs = [
        get_single_config(group_name, option_name) for option_name in group_options
    ]
    return group_configs


class TestGroupItem:
    """Base class to test a single group item, e.g. a single dataset or estimator."""

    @staticmethod
    def get_cfg(cfg: DictConfig) -> Optional[DictConfig]:
        return cfg


def generate_group_tests(group_name: str, metafunc):
    """Function to be used with `pytest_generate_tests` in order togenerate Pytest
    tests dynamically."""

    all_cfgs = get_group_configs(group_name)

    pytest_ids = []
    argvalues = []

    for cfg in all_cfgs:
        cfg = metafunc.cls.get_cfg(cfg)
        if cfg is not None:
            assert (
                hasattr(cfg, "name") and cfg.name is not None
            ), "group item has no name"
            pytest_ids.append(cfg.name)
            argvalues.append([cfg])

    metafunc.parametrize(["cfg"], argvalues, ids=pytest_ids, scope="class")
