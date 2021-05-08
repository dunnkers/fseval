import logging
from dataclasses import dataclass
from time import time

import numpy as np
import sklearn
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.base import is_classifier, is_regressor
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import log_loss, mean_absolute_error
from wandb.sklearn import (
    plot_feature_importances,
    plot_precision_recall,
    plot_roc,
    plot_summary_metrics,
)

from fseval.base import Configurable
from fseval.config import ExperimentConfig, ResampleConfig
from fseval.cv import CrossValidator
from fseval.datasets import Dataset
from fseval.rankers import Ranker
from fseval.resampling import Resample

logger = logging.getLogger(__name__)


@dataclass
class Experiment(ExperimentConfig, Configurable):
    def run(self):
        # load dataset and do cross-validation split
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
        ranker_log = {"ranker_fit_time": end_time - start_time}
        X_importances = self.dataset.get_feature_importances()
        if X_importances is None:
            pass
        elif np.ndim(X_importances) == 2:
            logger.warn("instance-based feature importance scores not supported yet.")
        else:
            # mean absolute error
            y_true = X_importances
            y_pred = ranking
            mae = mean_absolute_error(y_true, y_pred)
            ranker_log["ranker_mean_absolute_error"] = mae
            logger.info(f"{self.ranker.name} mean absolute error: {mae}")

            # log loss
            y_true = X_importances > 0
            y_pred = ranking
            log_loss_score = log_loss(y_true, y_pred, labels=[0, 1])
            ranker_log["ranker_log_loss"] = log_loss_score
            logger.info(f"{self.ranker.name} log loss score: {log_loss_score}")
        wandb.log(ranker_log)

        # validation
        n, p = X_train.shape
        k_best = np.arange(min(p, 50)) + 1
        best_score = 0
        for k in k_best:
            score = self.validator.select_fit_score(
                X_train,
                X_test,
                y_train,
                y_test,
                ranking,
                k,
            )

            # TODO this can be better implemented
            assert is_classifier(self.validator) or is_regressor(self.validator)
            if is_classifier(self.validator):
                wandb.log({"validator_p": k, "validation_accuracy": score})
            elif is_regressor(self.validator):
                wandb.log({"validator_p": k, "validation_r2_score": score})

            # save best score in wandb summary
            if score > best_score:
                best_score = score
                wandb.run.summary["best_validator_score"] = best_score
                wandb.run.summary["best_k"] = k
        logger.info(
            f"{self.validator.name} validation summary: {wandb.run.summary._as_dict()}"
        )
        wandb.finish()
