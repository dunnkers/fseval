import logging
from dataclasses import dataclass
from time import time
from typing import Any, List, cast

import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import log_loss, mean_absolute_error

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

        all_scores = []
        bootstraps = list(range(self.n_bootstraps))
        p = cast(int, self.dataset.p)
        k_best = np.arange(min(p, 50)) + 1
        for i in bootstraps:
            logger.info(f"running bootstrap #{i}: resample.random_state={i}")

            # resample
            self.resample.random_state = i
            X_train, X_test, y_train, y_test = data
            X_train, y_train = self.resample.transform(X_train, y_train)

            # feature ranking
            data = (X_train, X_test, y_train, y_test)
            ranking, fit_time = feature_ranker.run(data, callback_list)
            logger.info(f"{self.ranker.name} ranking fit time: %s", fit_time)
            ranker_log = {"ranker_fit_time": fit_time, "resample.random_state": i}

            # feature importances
            X_importances = self.dataset.get_feature_importances()
            if X_importances is None:
                pass
            elif np.ndim(X_importances) == 2:
                logger.warn(
                    "instance-based feature importance scores not supported yet."
                )
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

            callback_list.on_log(ranker_log)

            # validation
            scores = []
            for k in k_best:
                selector = SelectKBest(score_func=lambda *_: ranking, k=k)
                X_train_selected = selector.fit_transform(X_train, y_train)
                X_test_selected = selector.fit_transform(X_test, y_test)
                data = (X_train_selected, X_test_selected, y_train, y_test)

                score, fit_time = run_estimator.run(data, callback_list)
                logger.info(
                    f"{self.estimator.name} score: %s (fit time: %s)", score, fit_time
                )
                callback_list.on_log(
                    {
                        "estimator_score": score,
                        "estimator_fit_time": fit_time,
                        "k": k,
                        "resample.random_state": i,
                    },
                    commit=False,
                )
                scores.append(score)
            all_scores.append(scores)

        # average score
        avg_estimator_scores = np.mean(all_scores, axis=0)
        logger.info(f"→ average score = %s", avg_estimator_scores)
        callback_list.on_log(
            {"avg_estimator_scores": avg_estimator_scores, "bootstrap": bootstraps}
        )

        # summary
        best_k_index = np.argmax(avg_estimator_scores)
        best_k = k_best[best_k_index]
        callback_list.on_summary(
            {"best_k": best_k, "best_k_score": avg_estimator_scores[best_k_index]}
        )
