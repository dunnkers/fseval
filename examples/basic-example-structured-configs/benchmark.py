import hydra
from fseval.adapters.openml import OpenMLDataset
from fseval.config import DatasetConfig, EstimatorConfig, PipelineConfig
from fseval.main import run_pipeline
from fseval.types import Task
from hydra.core.config_store import ConfigStore
from sklearn.base import BaseEstimator
from sklearn.feature_selection import f_classif, mutual_info_classif

cs = ConfigStore.instance()

### 📈  Define a Feature Ranker, Anova F Value
class ANOVAFValueClassifier(BaseEstimator):
    def fit(self, X, y):
        scores, _ = f_classif(X, y)
        self.feature_importances_ = scores


anova_ranker_clf = EstimatorConfig(
    name="Anova F-Value",
    estimator={"_target_": "benchmark.ANOVAFValueClassifier"},
    _estimator_type="classifier",
    estimates_feature_importances=True,
)
cs.store(group="ranker", name="anova_ranker_clf", node=anova_ranker_clf)

### 📈  Define a Feature Ranker, MutualInfo
class MutualInfoClassifier(BaseEstimator):
    def fit(self, X, y):
        scores = mutual_info_classif(X, y)
        self.feature_importances_ = scores


mutual_info_clf = EstimatorConfig(
    name="Mutual Info",
    estimator={
        "_target_": "benchmark.MutualInfoClassifier",
    },
    _estimator_type="classifier",
    multioutput=False,
    estimates_feature_importances=True,
)
cs.store(group="ranker", name="mutual_info_clf", node=mutual_info_clf)

### 🧾  Define validator
knn_estimator = EstimatorConfig(
    name="k-NN",
    estimator={"_target_": "sklearn.neighbors.KNeighborsClassifier"},
    _estimator_type="classifier",
    estimates_target=True,
)

cs.store(group="validator", name="knn", node=knn_estimator)


### 💾  Define datasets
cs.store(
    group="dataset",
    name="iris",
    node=DatasetConfig(
        name="iris",
        task=Task.classification,
        adapter=OpenMLDataset(dataset_id=61, target_column="class"),
    ),
)

cs.store(
    group="dataset",
    name="ozone",
    node=DatasetConfig(
        name="Ozone Levels",
        task=Task.classification,
        adapter=OpenMLDataset(dataset_id=1487, target_column="Class"),
    ),
)


### ⚙️  Define pipeline config
cs.store(name="my_config", node=PipelineConfig())


### 🚀  Run fseval
@hydra.main(config_path=None, config_name="my_config")
def main(cfg: PipelineConfig) -> None:
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
