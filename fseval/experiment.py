import sklearn
from dataclasses import dataclass
from fseval.config import ExperimentConfig, ResampleConfig
from fseval.datasets import Dataset
from fseval.cv import CrossValidator
from fseval.resampling import Resample
from fseval.rankers import Ranker
from fseval.base import Configurable
from sklearn.feature_selection import SelectKBest
from hydra.utils import instantiate
from typing import Tuple, List
import numpy as np
from time import time
from omegaconf import OmegaConf
import wandb
from wandb.sklearn import (
    plot_feature_importances,
    plot_summary_metrics,
    plot_roc,
    plot_precision_recall,
)
import logging

logger = logging.getLogger(__name__)


@dataclass
class Experiment(ExperimentConfig, Configurable):
    def run(self):
        self.dataset.load()
        train_index, test_index = self.cv.get_split(self.dataset.X)
        train_index = self.resample.transform(train_index)
        X_train, X_test, y_train, y_test = self.dataset.get_subsets(
            train_index, test_index
        )

        # perform feature ranking
        wandb.init(project=self.project, config=self.get_config())
        start_time = time()
        self.ranker.fit(X_train, y_train)
        end_time = time()
        ranking = self.ranker.feature_importances_
        logger.info(f"{self.ranker.name} feature ranking: {ranking}")
        # log using wandb
        plot_feature_importances(self.ranker)
        if self.dataset.relevant_features is not None:
            assert y_true is not None, "not implemented yet"
            wandb.log(
                {
                    "ranker_log_loss": self.ranker.score(X=None, y=y_true),
                    "ranker_fit_time": end_time - start_time,
                }
            )

        # validation
        n, p = X_train.shape
        k_best = np.arange(min(p, 50)) + 1
        best_score = 0
        for k in k_best:
            self.validator.select_fit_score(
                X_train,
                X_test,
                y_train,
                y_test,
                ranking,
                k,
            )
            wandb.log({"validator_p": k, "validator_score": score})
            if score > best_score:
                best_score = score
                wandb.run.summary["best_validator_score"] = best_score
                wandb.run.summary["best_k"] = k
        logger.info(
            f"{self.validator.name} validation summary: {wandb.run.summary._as_dict()}"
        )
        wandb.finish()
