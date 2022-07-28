from typing import Callable, List, Optional, Tuple

from hydra import compose, initialize_config_module
from hydra.core.config_loader import ConfigLoader
from hydra.core.global_hydra import GlobalHydra
from hydra.core.object_type import ObjectType

from fseval.config import PipelineConfig

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
        initialize_config_module(
            config_module=config_module, job_name=job_name, version_base="1.1"
        )


def get_config(
    config_module: str, config_name: str, overrides: List[str] = []
) -> PipelineConfig:
    """Gets the fseval configuration as composed by Hydra. Local .yaml configuration
    and defaults are automatically merged."""
    _ensure_hydra_initialized(config_module)
    config = compose(config_name=config_name, overrides=overrides)
    return config  # type: ignore


def _get_config_loader(config_module: str) -> ConfigLoader:
    """Grabs the Hydra `ConfigLoader`."""
    _ensure_hydra_initialized(config_module)
    gh = GlobalHydra()
    cl = gh.config_loader()
    return cl


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


def get_group_pipeline_configs(
    config_module: str, config_name: str, group_name: str, should_test: Callable
) -> Tuple[List[PipelineConfig], List[str]]:
    group_options: List[str] = get_group_options(config_module, group_name)

    argvalues: List[PipelineConfig] = []
    pytest_ids: List[str] = []

    for group_option in group_options:
        cfg: PipelineConfig = get_config(
            config_module, config_name, overrides=[f"{group_name}={group_option}"]
        )

        should_test_option: bool = should_test(cfg, group_name=group_name)
        if should_test_option:
            pytest_ids.append(group_option)
            argvalues.append(cfg)

    return argvalues, pytest_ids
