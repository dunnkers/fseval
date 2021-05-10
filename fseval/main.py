import hydra
import wandb
from hydra.utils import instantiate

from fseval.config import BaseConfig


@hydra.main(config_path="conf", config_name="my_config")
def main(cfg: BaseConfig) -> None:
    cfg.pipeline.dataset = cfg.dataset
    cfg.pipeline.cv = cfg.cv
    cfg.pipeline.resample = cfg.resample
    instance = instantiate(cfg)

    pass
    # start wandb right away to capture any errors to its logging system
    # wandb.init(project=cfg.project, config=experiment.get_config())

    # experiment.run()

    # wandb.finish()


if __name__ == "__main__":
    main()
