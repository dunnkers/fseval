from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from logging import Logger, getLogger
from time import time
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from fseval.base import Configurable
from fseval.cv import CrossValidator
from fseval.datasets import Dataset
from fseval.resampling import Resample

from ._callbacks import CallbackList


class PipelineComponent(ABC, Configurable):
    logger: Logger = getLogger(__name__)

    @abstractmethod
    def run(self, args: Any, callback_list: CallbackList) -> Any:
        ...


@dataclass
class SubsetLoaderPipe(PipelineComponent):
    dataset: Dataset
    cv: CrossValidator

    def run(self, args: Any, callback_list: CallbackList) -> Any:
        # load dataset
        self.dataset.load()

        # send runtime properties to callbacks
        callback_list.on_pipeline_config_update(
            {
                "dataset": {
                    "n": self.dataset.n,
                    "p": self.dataset.p,
                    "multivariate": self.dataset.multivariate,
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
class ResamplerPipe(PipelineComponent):
    resample: Resample

    def run(self, args: Any, callback_list: CallbackList) -> Any:
        output = self.resample.transform(input)
        return output


@dataclass
class FeatureRankingPipe(PipelineComponent):
    ranker: Any = None

    def run(self, args: Any, callback_list: CallbackList) -> Any:
        X_train, _, y_train, _ = args

        # run feature ranking
        start_time = time()
        self.ranker.estimator.fit(X_train, y_train)
        end_time = time()

        # metrics
        ranking = self.ranker.estimator.feature_importances_
        ranking = np.asarray(ranking)
        ranking /= sum(ranking)
        fit_time = end_time - start_time

        # call callback on file save
        ranking_df = pd.DataFrame(ranking)
        ranking_csv = ranking_df.to_csv(index=False)
        callback_list.on_file_save("ranking", ranking_csv)

        return ranking, fit_time


@dataclass
class RunEstimatorPipe(PipelineComponent):
    estimator: Any = None

    def run(self, args: Any, callback_list: CallbackList) -> Any:
        X_train, X_test, y_train, y_test = args

        # run estimator
        start_time = time()
        self.estimator.estimator.fit(X_train, y_train)
        end_time = time()

        # metrics
        score = self.estimator.estimator.score(X_test, y_test)
        fit_time = end_time - start_time

        return score, fit_time
