from fseval.config import ExperimentConfig
from hydra.utils import instantiate

class Experiment:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.dataset    = instantiate(cfg.dataset)
        self.cv         = instantiate(cfg.cv)
        self.ranker     = instantiate(cfg.ranker)

    def run(self):
        print(self.cfg)
        print(self.dataset)
        print(self.cv)