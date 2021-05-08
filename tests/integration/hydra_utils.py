from typing import List, Optional, cast

from hydra import compose, initialize_config_module
from hydra.core.config_loader import ConfigLoader
from hydra.core.global_hydra import GlobalHydra
from hydra.core.object_type import ObjectType
from hydra.types import RunMode
from omegaconf import DictConfig, OmegaConf


def _ensure_hydra_initialized() -> None:
    gh = GlobalHydra()
    if not gh.is_initialized():
        initialize_config_module(config_module="fseval.conf", job_name="fseval_tests")


def _get_config_loader() -> ConfigLoader:
    _ensure_hydra_initialized()
    gh = GlobalHydra()
    cl = gh.config_loader()
    return cl


def get_config() -> DictConfig:
    _ensure_hydra_initialized()
    config = compose(config_name="config")
    return config


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


class TestGroupItem:
    @staticmethod
    def get_cfg(cfg: DictConfig) -> DictConfig:
        return cfg


def generate_group_tests(group_name: str, metafunc):
    all_cfgs = get_group_configs(group_name)

    pytest_ids = []
    argvalues = []

    for cfg in all_cfgs:
        cfg = metafunc.cls.get_cfg(cfg)
        if cfg:
            assert (
                hasattr(cfg, "name") and cfg.name is not None
            ), "group item has no name"
            pytest_ids.append(cfg.name)
            argvalues.append([cfg])

    metafunc.parametrize(["cfg"], argvalues, ids=pytest_ids, scope="class")
