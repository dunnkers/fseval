import hydra
from fseval.adapters.openml import OpenMLDataset
from fseval.config import (
    DatasetConfig,
    EstimatorConfig,
    PipelineConfig,
    TaskedEstimatorConfig,
)
from fseval.main import run_pipeline
from fseval.types import Task
from hydra.core.config_store import ConfigStore
from sklearn.base import BaseEstimator
from sklearn.feature_selection import f_classif

cs = ConfigStore.instance()

### 📈  Define Feature Ranker
class ANOVAFValueClassifier(BaseEstimator):
    def fit(self, X, y):
        scores, _ = f_classif(X, y)
        self.feature_importances_ = scores


anova_ranker = TaskedEstimatorConfig(
    name="Anova F-Value",
    classifier=EstimatorConfig(
        estimator={"_target_": "somebenchmark.ANOVAFValueClassifier"}
    ),
    estimates_feature_importances=True,
)

cs.store(group="ranker", name="anova_f_value", node=anova_ranker)

### 🧾  Define validator
knn_estimator = TaskedEstimatorConfig(
    name="k-NN",
    classifier=EstimatorConfig(
        estimator={"_target_": "sklearn.neighbors.KNeighborsClassifier"}
    ),
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
