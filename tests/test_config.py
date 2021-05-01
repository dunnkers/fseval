from omegaconf import OmegaConf, DictConfig


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
