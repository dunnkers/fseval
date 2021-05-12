import logging
from dataclasses import dataclass
from time import time
from typing import Any, List

import numpy as np

from ._callbacks import CallbackList
from ._components import RunEstimatorPipe, SubsetLoaderPipe
from ._pipeline import Pipeline

logger = logging.getLogger(__name__)


@dataclass
class RunEstimator(Pipeline):
    estimator: Any = None

    def run(self, input: Any, callback_list: CallbackList) -> Any:
        subset_loader = SubsetLoaderPipe(self.dataset, self.cv)
        data = subset_loader.run(None, callback_list)

        run_estimator = RunEstimatorPipe(self.estimator)
        score, fit_time = run_estimator.run(data, callback_list)

        logger.info(f"{self.estimator.name} score: {score}")
        callback_list.on_log({"estimator_score": score, "estimator_fit_time": fit_time})
