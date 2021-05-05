import hydra
from dataclasses import dataclass
from hydra.utils import instantiate
from fseval.config import ExperimentConfig


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: ExperimentConfig) -> None:
    experiment = instantiate(cfg)
    experiment.run()


if __name__ == "__main__":
    main()
