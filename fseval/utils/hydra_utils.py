from typing import Callable, List, Optional, Tuple

from fseval.config import PipelineConfig
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


def _ensure_hydra_initialized(
    config_module: str, job_name: str = "fseval_hydra_utils"
) -> None:
    """Initializes Hydra only if it is not already initialized."""
    gh = GlobalHydra()
    if not gh.is_initialized():
        initialize_config_module(config_module=config_module, job_name=job_name)


def _get_config_loader(config_module: str) -> ConfigLoader:
    """Grabs the Hydra `ConfigLoader`."""
    _ensure_hydra_initialized(config_module)
    gh = GlobalHydra()
    cl = gh.config_loader()
    return cl


def get_config(
    config_module: str, config_name: str, overrides: List[str] = []
) -> PipelineConfig:
    """Gets the fseval configuration as composed by Hydra. Local .yaml configuration
    and defaults are automatically merged."""
    _ensure_hydra_initialized(config_module)
    config = compose(config_name=config_name, overrides=overrides)
    return config  # type: ignore


def get_single_config(
    config_module: str,
    config_name: str,
    group_name: str,
    option_name: str,
    overrides: List[str] = [],
) -> DictConfig:
    """Returns a single config, e.g. `ranker/chi2` with the structured config
    defined in `fseval/config.py` merged into it."""
    cl = _get_config_loader(config_module)
    cfg = OmegaConf.create()

    structured_config = cl.load_configuration(config_name, overrides, RunMode.RUN)
    OmegaConf.set_struct(structured_config, False)
    cfg.merge_with(structured_config)

    item_config = cl.load_configuration(f"{group_name}/{option_name}", [], RunMode.RUN)
    cfg.merge_with(item_config)

    return cfg[group_name]


def get_group_options(
    config_module: str,
    group_name: str,
    results_filter: Optional[ObjectType] = ObjectType.CONFIG,
) -> List[str]:
    """Gets the options for a certain grouop.

    e.g. `get_group_options(<dataset_name>)` returns a list with all
    available dataset names for use in fseval."""
    cl = _get_config_loader(config_module)
    group_options = cl.get_group_options(group_name)
    return group_options


def get_group_configs(
    config_module: str,
    config_name: str,
    group_name: str,
    overrides: List[str] = [],
    results_filter: Optional[ObjectType] = ObjectType.CONFIG,
) -> List[DictConfig]:
    """Gets the configs related to a certain group, as `DictConfig`'s."""
    group_options = get_group_options(config_module, group_name)
    group_configs = [
        get_single_config(config_module, config_name, group_name, option_name)
        for option_name in group_options
    ]
    return group_configs


def generate_group_tests(
    config_module: str, config_name: str, group_name: str, metafunc
):
    """Function to be used with `pytest_generate_tests` in order togenerate Pytest
    tests dynamically."""

    all_cfgs = get_group_configs(config_module, config_name, group_name)

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


def get_group_pipeline_configs(
    config_module: str, config_name: str, group_name: str, should_test: Callable
) -> Tuple[List[List[PipelineConfig]], List[str]]:
    group_options: List[str] = get_group_options(config_module, group_name)

    argvalues: List[List[PipelineConfig]] = []
    pytest_ids: List[str] = []

    for group_option in group_options:
        cfg: PipelineConfig = get_config(
            config_module, config_name, overrides=[f"{group_name}={group_option}"]
        )

        should_test_option: bool = should_test(cfg, group_name=group_name)
        if should_test_option:
            pytest_ids.append(group_option)
            argvalues.append([cfg])

    return argvalues, pytest_ids
