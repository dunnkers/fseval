from fseval.config import ExperimentConfig
from hydra.utils import instantiate
from typing import Tuple, List


class Experiment:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.project = cfg.project
        self.cv_fold = cfg.cv_fold
        self.dataset = instantiate(cfg.dataset)
        self.cv = instantiate(cfg.cv)
        self.ranker = instantiate(cfg.ranker)
        self.validator = instantiate(cfg.validator)

    def get_splits(self, X) -> Tuple[List, List]:
        splits = list(self.cv.split(X))
        train_index, test_index = splits[self.cv_fold]
        return train_index, test_index

    def run(self):
        print(self.cfg)
        print(self.dataset)
        print(self.cv)
