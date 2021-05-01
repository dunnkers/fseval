from hydra.experimental import initialize, compose
from hydra.core.config_store import ConfigStore
from fseval.config import ExperimentConfig
from fseval.experiment import Experiment
from omegaconf import OmegaConf


def test_config_loading() -> None:
    cs = ConfigStore.instance()
    cs.store(name='config', node=ExperimentConfig)

    with initialize(config_path='../conf'):
        # config is relative to a module
        cfg: ExperimentConfig = compose(config_name='config') # type: ignore
        assert cfg.project is not None
        assert cfg.datasrc is not None
        assert cfg.cv is not None
        assert cfg.bootstrap is not None

        experiment = Experiment(cfg)
        assert experiment is not None