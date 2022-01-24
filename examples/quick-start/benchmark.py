import hydra
from sklearn.base import BaseEstimator
from sklearn.feature_selection import f_classif, mutual_info_classif

from fseval.config import PipelineConfig
from fseval.main import run_pipeline


class ANOVAFValueClassifier(BaseEstimator):
    def fit(self, X, y):
        scores, _ = f_classif(X, y)
        self.feature_importances_ = scores


class MutualInfoClassifier(BaseEstimator):
    def fit(self, X, y):
        scores = mutual_info_classif(X, y)
        self.feature_importances_ = scores


@hydra.main(config_path="conf", config_name="my_config")
def main(cfg: PipelineConfig) -> None:
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
