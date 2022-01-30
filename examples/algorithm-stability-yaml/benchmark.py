from typing import Dict

import hydra
from fseval.config import PipelineConfig
from fseval.main import run_pipeline
from fseval.types import AbstractEstimator, AbstractMetric, Callback
from sklearn.base import BaseEstimator
from sklearn.feature_selection import f_classif, mutual_info_classif


class ANOVAFValueClassifier(BaseEstimator):
    def fit(self, X, y):
        scores, _ = f_classif(X, y)
        self.feature_importances_ = scores


class MutualInfoClassifier(BaseEstimator):
    def fit(self, X, y):
        scores = mutual_info_classif(X, y)
        self.feature_importances_ = scores


class StabilityNogueira(AbstractMetric):
    def score_bootstrap(
        self,
        ranker: AbstractEstimator,
        validator: AbstractEstimator,
        callbacks: Callback,
        scores: Dict,
        **kwargs,
    ) -> Dict:
        ...
        print("end", scores["feature_importances"])
        return {}


@hydra.main(config_path="conf", config_name="my_config")
def main(cfg: PipelineConfig) -> None:
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
