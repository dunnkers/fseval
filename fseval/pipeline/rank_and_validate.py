import logging
from dataclasses import dataclass
from time import time
from typing import Any, List

import numpy as np

from ._callbacks import CallbackList
from ._pipeline import Pipeline
from .feature_ranking import FeatureRanking
from .run_estimator import RunEstimator

logger = logging.getLogger(__name__)


@dataclass
class RankAndValidate(Pipeline):
    ranker: Any = None
    estimator: Any = None

    def run(self, input: Any, callback_list: CallbackList) -> Any:
        pass
