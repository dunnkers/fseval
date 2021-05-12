import logging
from dataclasses import dataclass
from time import time
from typing import Any, List

import numpy as np
import pandas as pd

from ._callbacks import CallbackList
from ._components import FeatureRankingPipe, SubsetLoaderPipe
from ._pipeline import Pipeline

logger = logging.getLogger(__name__)


@dataclass
class FeatureRanking(Pipeline):
    ranker: Any = None

    def run(self, args: Any, callback_list: CallbackList) -> Any:
        subset_loader = SubsetLoaderPipe(self.dataset, self.cv)
        data = subset_loader.run(None, callback_list)

        feature_ranker = FeatureRankingPipe(self.ranker)
        ranking, fit_time = feature_ranker.run(data, callback_list)

        logger.info(f"{self.ranker.name} feature ranking: {ranking}")
        callback_list.on_log({"ranker_fit_time": fit_time})
