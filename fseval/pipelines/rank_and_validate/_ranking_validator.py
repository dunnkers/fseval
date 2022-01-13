from dataclasses import dataclass
from logging import Logger, getLogger
from typing import Dict, Union

import numpy as np
import pandas as pd
from omegaconf import MISSING

from fseval.types import IncompatibilityError, TerminalColor

from .._experiment import Experiment
from ._config import RankAndValidatePipeline


@dataclass
class RankingValidator(Experiment, RankAndValidatePipeline):
    """Validates a feature ranking. A feature ranking is validated by comparing the
    estimated feature- ranking, importance or support to the ground truth feature
    importances. Generally, the ground-truth feature importances are only available
    when a dataset is synthetically generated."""

    bootstrap_state: int = MISSING

    logger: Logger = getLogger(__name__)

    def __post_init__(self):
        if not (
            self.ranker.estimates_feature_importances
            or self.ranker.estimates_feature_ranking
            or self.ranker.estimates_feature_support
        ):
            raise IncompatibilityError(
                f"{self.ranker.name} performs no form of feature ranking: "
                + "this estimator cannot be used as a ranker."
            )

        super(RankingValidator, self).__post_init__()

    @property
    def _cache_filename(self):
        override = f"bootstrap_state={self.bootstrap_state}"
        filename = f"ranking[{override}].pickle"

        return filename

    def _get_estimator(self):
        yield self.ranker

    def prefit(self):
        self.ranker._load_cache(self._cache_filename, self.storage)

    def fit(self, X, y):
        self.logger.info(f"fitting ranker: " + TerminalColor.yellow(self.ranker.name))

        super(RankingValidator, self).fit(X, y)

    def postfit(self):
        self.ranker._save_cache(self._cache_filename, self.storage)

    def score(self, X, y, **kwargs) -> Union[Dict, pd.DataFrame, np.generic, None]:
        """Scores a feature ranker, if a ground-truth on the desired dataset
        feature importances is available. If this is the case, the estimated normalized
        feature importances are compared to the desired ones using two metrics:
        log loss and the R^2 score. Whilst the log loss converts the ground-truth
        desired feature rankings to a binary value, 0/1, the R^2 score always works."""

        # ensure ground truth feature_importances are 1-dimensional
        feature_importances = kwargs.pop("feature_importances", None)
        if feature_importances is not None:
            assert (
                np.ndim(feature_importances) == 1
            ), "instance-based not supported yet."

        # add fitting time and bootstrap to score
        scores_dict = {
            "fit_time": self.ranker.fit_time_,
            "bootstrap_state": self.bootstrap_state,
        }

        # create dataframe
        scores = pd.DataFrame([scores_dict])

        # add custom metrics
        for metric_name, metric_class in self.metrics.items():
            scores_metric = metric_class.score_ranking(
                scores,
                self.ranker,
                self.bootstrap_state,
                self.callbacks,
                feature_importances,
            )

            if scores_metric is not None:
                scores = scores_metric

        return scores
