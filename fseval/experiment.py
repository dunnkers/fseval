import sklearn
from fseval.config import ExperimentConfig, ResampleConfig
from fseval.dataset import Dataset
from fseval.cv import CrossValidator
from fseval.ranker import Ranker
from sklearn.base import BaseEstimator
from hydra.utils import instantiate
from typing import Tuple, List
import numpy as np
from omegaconf import OmegaConf


class Experiment:
    def __init__(self, cfg: ExperimentConfig):
        # constants
        self.cfg = cfg
        self.project = cfg.project
        self.resample: ResampleConfig = cfg.resample

        # create instances using _target_ properties
        self.dataset: Dataset = instantiate(cfg.dataset)
        self.cv: CrossValidator = instantiate(cfg.cv)
        self.ranker: Ranker = instantiate(cfg.ranker)
        self.validator: BaseEstimator = instantiate(cfg.validator)

    def run(self):
        X, y = self.dataset.load()
        train_index, test_index = self.cv.get_split(X)

        # resample
        resample_cfg = OmegaConf.to_container(self.resample)
        train_index = sklearn.utils.resample(train_index, **resample_cfg)

        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        # perform feature ranking
        n, p = X_train.shape
        print(f"Feature ranking with (n={n}, p={p}).")
        self.ranker.fit(X_train, y_train)
        ranking = self.ranker.feature_importances_
        ranking = ranking / np.sum(ranking)  # normalize as probability vector
        print(ranking)
