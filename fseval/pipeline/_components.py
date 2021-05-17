import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import time
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from fseval.cv import CrossValidator
from fseval.datasets import Dataset
from fseval.resampling import Resample
from sklearn.metrics import log_loss, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from ._callbacks import CallbackList


class Pipe(ABC):
    callback_list: CallbackList

    @abstractmethod
    def run(self) -> Any:
        ...


@dataclass
class SubsetLoaderPipe(Pipe):
    dataset: Dataset
    cv: CrossValidator

    def run(self) -> Any:
        # load dataset
        self.dataset.load()

        # send runtime properties to callbacks
        self.callback_list.on_pipeline_config_update(
            {
                "dataset": {
                    "n": self.dataset.n,
                    "p": self.dataset.p,
                    "multioutput": self.dataset.multioutput,
                }
            }
        )

        # cross-validation split
        train_index, test_index = self.cv.get_split(self.dataset.X)

        # cross-validation subsets
        X_train, X_test, y_train, y_test = self.dataset.get_subsets(
            train_index, test_index
        )

        return X_train, X_test, y_train, y_test


@dataclass
class ResamplerPipe(Pipe):
    resample: Resample

    def run(self, *args: Any, **kwargs: Any) -> Any:
        output = self.resample.transform(input)
        return output


@dataclass
class FeatureRankingPipe(Pipe):
    dataset: Dataset
    ranker: Any
    resample: Resample

    def run(self, *args: Any, **kwargs: Any) -> Any:
        X_train, _, y_train, _ = args
        self.callback_list.on_log(f"feature ranking (resample.random_state={i})")

        # resample

        # run feature ranking
        start_time = time()
        self.ranker.fit(X_train, y_train)
        end_time = time()

        # metrics
        ranking = self.ranker.feature_importances_
        ranking = np.asarray(ranking)
        ranking /= sum(ranking)
        fit_time = end_time - start_time

        # call callback on file save
        ranking_df = pd.DataFrame(ranking)
        ranking_csv = ranking_df.to_csv(index=False)
        self.callback_list.on_file_save("ranking.csv", ranking_csv)

        # feature importances
        X_importances = self.dataset.get_feature_importances()
        if X_importances is not None:
            assert np.ndim(X_importances) == 1, "instance-based not supported yet."

            # mean absolute error
            y_true = X_importances
            y_pred = ranking
            mae = mean_absolute_error(y_true, y_pred)
            ranker_log["ranker_mean_absolute_error"] = mae

            # log loss
            y_true = X_importances > 0
            y_pred = ranking
            log_loss_score = log_loss(y_true, y_pred, labels=[0, 1])
            ranker_log["ranker_log_loss"] = log_loss_score

        return ranking, fit_time


@dataclass
class RunEstimatorPipe(Pipe):
    estimator: Any = None

    def run(self, *args: Any, **kwargs: Any) -> Any:
        X_train, X_test, y_train, y_test = args
        # run estimator
        start_time = time()
        self.estimator.fit(X_train, y_train)
        end_time = time()

        # metrics
        score = self.estimator.score(X_test, y_test)
        fit_time = end_time - start_time

        return pd.DataFrame([{"score": score, "fit_time": fit_time}])


@dataclass
class RankingValidator(Pipe):
    run_estimator: Pipe

    def run(self, X_train, X_test, y_train, y_test, ranking):
        p = X_train.shape[1]
        k_best = np.arange(min(p, 50)) + 1
        results = []

        is_slurm = os.environ.get("SLURM_JOB_ID", None) is not None
        progbar = tqdm(k_best, disable=is_slurm, desc="validating ranking")

        for k in progbar:
            selector = SelectKBest(score_func=lambda *_: ranking, k=k)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.fit_transform(X_test, y_test)
            data = (X_train_selected, X_test_selected, y_train, y_test)

            score, fit_time = self.run_estimator.run(*data)
            results.append(
                {
                    "k": k,
                    "estimator_score": score,
                    "estimator_fit_time": fit_time,
                }
            )

        df = pd.DataFrame(results)
        return df
