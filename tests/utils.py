from typing import List, Optional, cast

from fseval.config import ExperimentConfig
from hydra import compose, initialize
from hydra.core.config_loader import ConfigLoader
from hydra.core.global_hydra import GlobalHydra
from hydra.core.object_type import ObjectType
from hydra.types import RunMode
from omegaconf import DictConfig, OmegaConf


def _get_config_loader() -> ConfigLoader:
    gh = GlobalHydra()
    if not gh.is_initialized():
        initialize(config_path="../fseval/conf")
        # compose(config_name="config")
    cl = gh.config_loader()
    return cl


def get_single_config(group_name: str, option_name: str) -> DictConfig:
    """Returns a single config, e.g. `ranker/chi2` with the structured config
    defined in `fseval/config.py` merged into it."""
    cl = _get_config_loader()
    cfg = OmegaConf.create()

    structured_config = cl.load_configuration("base_config", [], RunMode.RUN)
    cfg.merge_with(structured_config)

    item_config = cl.load_configuration(f"{group_name}/{option_name}", [], RunMode.RUN)
    cfg.merge_with(item_config)

    return cfg[group_name]


def get_group_options(
    group_name: str, results_filter: Optional[ObjectType] = ObjectType.CONFIG
) -> List[str]:
    cl = _get_config_loader()
    group_options = cl.get_group_options(group_name)
    return group_options


def get_group_configs(
    group_name: str, results_filter: Optional[ObjectType] = ObjectType.CONFIG
) -> List[DictConfig]:
    group_options = get_group_options(group_name)
    group_configs = [
        get_single_config(group_name, option_name) for option_name in group_options
    ]
    return group_configs
