import sklearn
from dataclasses import dataclass
from fseval.config import ExperimentConfig, ResampleConfig
from fseval.datasets import Dataset
from fseval.cv import CrossValidator
from fseval.resampling import Resample
from fseval.rankers import Ranker
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectKBest
from hydra.utils import instantiate
from typing import Tuple, List
import numpy as np
from omegaconf import OmegaConf
import wandb


@dataclass
class Experiment(ExperimentConfig, BaseEstimator):
    def run(self):
        self.dataset.load()
        train_index, test_index = self.cv.get_split(self.dataset.X)
        train_index = self.resample.transform(train_index)
        X_train, X_test, y_train, y_test = self.dataset.get_subsets(
            train_index, test_index
        )

        # perform feature ranking
        wandb.init(project=self.project, config=self.get_params())
        self.ranker.fit(X_train, y_train)
        ranking = self.ranker.feature_importances_
        print(ranking)

        # validation
        n, p = X_train.shape
        for k in np.arange(min(p, 100)) + 1:
            selector = SelectKBest(score_func=lambda *_: ranking, k=k)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.fit_transform(X_test, y_test)

            self.validator.fit(X_train_selected, y_train)
            score = self.validator.score(X_test_selected, y_test)
            print(f"k={k} {score}")
