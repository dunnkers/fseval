import logging
import hydra
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from fseval.config import ExperimentConfig
from fseval.experiment import Experiment

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config")
def run(cfg: ExperimentConfig) -> None:
    experiment = instantiate(cfg)
    experiment.run()


if __name__ == "__main__":
    print("hydra version", hydra.__version__)
    logger.info(f"hydra version: {hydra.__version__}")
    run()
