import hydra
import numpy as np
from fseval.config import PipelineConfig
from fseval.main import run_pipeline
from infinite_selection import InfFS
from sklearn.base import BaseEstimator
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.preprocessing import minmax_scale
from stability_selection import StabilitySelection as RealStabilitySelection


class StabilitySelection(RealStabilitySelection):
    def fit(self, X, y):
        super(StabilitySelection, self).fit(X, y)
        self.support_ = self.get_support()
        self.feature_importances_ = np.max(self.stability_scores_, axis=1)


class InfiniteSelectionEstimator(BaseEstimator):
    def fit(self, X, y):
        inf = InfFS()
        [RANKED, WEIGHT] = inf.infFS(X, y, alpha=0.5, supervision=1, verbose=1)

        self.feature_importances_ = WEIGHT
        self.ranking_ = RANKED


class Chi2Classifier(BaseEstimator):
    def fit(self, X, y):
        X = minmax_scale(X)
        scores, _ = chi2(X, y)
        self.feature_importances_ = scores


class ANOVAFValueClassifier(BaseEstimator):
    def fit(self, X, y):
        scores, _ = f_classif(X, y)
        self.feature_importances_ = scores


class MutualInfoClassifier(BaseEstimator):
    def fit(self, X, y):
        scores = mutual_info_classif(X, y)
        self.feature_importances_ = scores


@hydra.main(config_path="conf", config_name="my_config", version_base="1.1")
def main(cfg: PipelineConfig) -> None:
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
