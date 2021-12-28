<p align="center">
  <img width="100%" src="./docs/header.png">
</p>

# fseval

[![build status](https://github.com/dunnkers/fseval/actions/workflows/python-app.yml/badge.svg)](https://github.com/dunnkers/fseval/actions/workflows/python-app.yml) [![pypi badge](https://img.shields.io/pypi/v/fseval.svg?maxAge=3600)](https://pypi.org/project/fseval/) [![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Downloads](https://pepy.tech/badge/fseval/month)](https://pepy.tech/project/fseval) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fseval) [![codecov](https://codecov.io/gh/dunnkers/fseval/branch/master/graph/badge.svg?token=R5ZXH8UPCI)](https://codecov.io/gh/dunnkers/fseval) [![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/dunnkers/fseval.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/dunnkers/fseval/context:python)

A Feature Ranker benchmarking pipeline. Useful for Feature Selection and Interpretable AI methods.


## Install

```shell
pip install fseval
```

## Usage
```python
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
```

Any [sklearn](https://scikit-learn.org/) style estimator can be used as a Feature Ranker. Estimator must estimate at least one of:

1. **Feature importance**, using `feature_importances_`.
2. **Feature subset**, using `feature_support_`.
3. **Feature ranking**, using `feature_ranking_`.

Main functionality:
- 📊 **Online dashboard**. Experiments can be uploaded to [wandb](https://wandb.ai) for seamless experiment tracking and visualization. Feature importance and subset validation plots are built-in. 
- 🔄 **Scikit-Learn integration**. Integrates nicely with [sklearn](https://scikit-learn.org/). Any estimator that implements `fit` is supported.
- 🗄 **Dataset adapters**. Datasets can be loaded dynamically using an _adapter_. [OpenML](https://www.openml.org/search?type=data) support is built-in.
- 🎛 **Synthetic dataset generation**. Synthetic datasets can be generated and configured right in the library itself.
- 📌 **Relevant features ground-truth**. Datasets can have ground-truth relevant features defined, so the estimated versus the ground-truth feature importance is automatically plotted in the dashboard.
- ⚜️ **Subset validation**. Allows you to validate the quality of a feature ranking, by running a _validation_ estimator on some of the `k` best feature subsets.
- ⚖️ **Bootstrapping**. Allows you to approximate the _stability_ of an algorithm by running multiple experiments on bootstrap resampled datasets.
- ⚙️ **Reproducible configs**. Uses [Hydra](https://hydra.cc/) as a config parser, to allow configuring every part of the experiment. The config can be uploaded to wandb, so the experiment can be replayed later.

## About
Built by at the University of Groningen.

---

<p align="center">2021 — <a href="https://dunnkers.com/">Jeroen Overschie</a></p>