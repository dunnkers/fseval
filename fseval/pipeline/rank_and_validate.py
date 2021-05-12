import logging
from dataclasses import dataclass
from time import time
from typing import Any, List

import numpy as np

from ._callbacks import CallbackList
from ._components import FeatureRankingPipe, RunEstimatorPipe, SubsetLoaderPipe
from ._pipeline import Pipeline
from .feature_ranking import FeatureRanking
from .run_estimator import RunEstimator

logger = logging.getLogger(__name__)


@dataclass
class RankAndValidate(Pipeline):
    ranker: Any = None
    estimator: Any = None
    n_bootstraps: int = 1

    def run(self, input: Any, callback_list: CallbackList) -> Any:
        subset_loader = SubsetLoaderPipe(self.dataset, self.cv)
        data = subset_loader.run(None, callback_list)

        feature_ranker = FeatureRankingPipe(self.ranker)
        run_estimator = RunEstimatorPipe(self.estimator)
        # validate_ranker = ValidateRankingPipe(
        #     self.estimator, n_bootstraps=self.n_bootstraps
        # )

        for i in range(self.n_bootstraps):
            logger.info(f"running bootstrap #{i}: resample.random_state={i}")

            # resample
            self.resample.random_state = i
            X_train, X_test, y_train, y_test = data
            X_train, y_train = self.resample.transform((X_train, y_train))

            # feature ranking
            data = (X_train, X_test, y_train, y_test)
            ranking, fit_time = feature_ranker.run(data, callback_list)
            logger.info("ranking fit time: %s", fit_time)

            k_best = np.arange(min(self.dataset.p, 50)) + 1
            for k in k_best:
                score, fit_time = run_estimator(data, callback_list)
                logger.info("estimator fit time: %s", fit_time)
