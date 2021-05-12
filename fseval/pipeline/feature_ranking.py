import logging
from dataclasses import dataclass
from time import time
from typing import Any, List

import numpy as np

from ._callbacks import CallbackList
from ._pipeline import Pipeline

logger = logging.getLogger(__name__)


@dataclass
class FeatureRanking(Pipeline):
    ranker: Any = None

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

        # feature ranking
        start_time = time()
        self.ranker.estimator.fit(X_train, y_train)
        end_time = time()
        ranking = self.ranker.estimator.feature_importances_
        ranking = np.asarray(ranking)
        ranking /= sum(ranking)

        logger.info(f"{self.ranker.name} feature ranking: {ranking}")
        callback_list.on_log({"ranker_fit_time": end_time - start_time})
