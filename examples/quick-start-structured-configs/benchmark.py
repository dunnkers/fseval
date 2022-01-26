import hydra
from conf.dataset.synthetic import synthetic_dataset
from conf.my_config import my_config
from conf.ranker.anova import anova_ranker
from conf.ranker.mutual_info import mutual_info_ranker
from conf.validator.knn import knn_validator
from hydra.core.config_store import ConfigStore
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


cs = ConfigStore.instance()
cs.store(group="dataset", name="synthetic", node=synthetic_dataset)
cs.store(group="ranker", name="anova", node=anova_ranker)
cs.store(group="ranker", name="mutual_info", node=mutual_info_ranker)
cs.store(group="validator", name="knn", node=knn_validator)
cs.store(name="my_config", node=my_config)


@hydra.main(config_path="conf", config_name="my_config")
def main(cfg: PipelineConfig) -> None:
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
