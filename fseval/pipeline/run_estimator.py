import logging
from dataclasses import dataclass
from time import time
from typing import Any, List

import numpy as np

from ._callbacks import CallbackList
from ._pipeline import Pipeline

logger = logging.getLogger(__name__)


@dataclass
class RunEstimator(Pipeline):
    estimator: Any = None

    def run(self, input: Any, callback_list: CallbackList) -> Any:
        # load dataset
        self.dataset.load()
        # send runtime properties to callbacks
        callback_list.on_config_update(
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

        # resampling; with- or without replacement
        train_index = self.resample.transform(train_index)

        # cross-validation subsets
        X_train, X_test, y_train, y_test = self.dataset.get_subsets(
            train_index, test_index
        )

        # run estimator
        start_time = time()
        self.estimator.estimator.fit(X_train, y_train)
        end_time = time()
        score = self.estimator.estimator.score(X_test, y_test)

        logger.info(f"{self.estimator.name} score: {score}")
        callback_list.on_log(
            {"estimator_score": score, "estimator_fit_time": end_time - start_time}
        )
