from fseval.config import ExperimentConfig
from hydra.utils import instantiate

class Experiment:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.datasrc = instantiate(cfg.datasrc)
        self.cv = instantiate(cfg.cv)

    def run(self):
        print(self.cfg)
        print(self.datasrc)
        print(self.cv)