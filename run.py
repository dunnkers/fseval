import hydra
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from fseval.config import ExperimentConfig
from fseval.experiment import Experiment

cs = ConfigStore.instance()
cs.store(name='config', node=ExperimentConfig)

@hydra.main(config_path='conf', config_name='config')
def run(cfg: ExperimentConfig) -> None:
    experiment = Experiment(cfg)
    experiment.run()

if __name__ == '__main__':
    run()
