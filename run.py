import hydra
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from fseval.config import ExperimentConfig
from fseval.experiment import Experiment


@hydra.main(config_path="conf", config_name="config")
def run(cfg: ExperimentConfig) -> None:
    experiment = instantiate(cfg)
    experiment.run()


if __name__ == "__main__":
    run()
