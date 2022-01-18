---
sidebar_position: 1
---

# Quick start

fseval helps you benchmark **Feature Selection** and **Feature Ranking** algorithms. Any algorithm that ranks features in importance.

It comes useful if you are one of the following types of users:
1. **Feature Selection algorithm authors**. You are the author of a novel Feature Selection algorithm. Now, you have to prove the performance of your algorithm against other competitors. Therefore, you are going to run a large-scale benchmark. Many authors, however, spend much time rewriting similar pipelines to benchmark their algorithms. fseval helps you run benchmarks in a structured manner, on supercomputer clusters or on the cloud.
2. **Feature Ranker algorithm authors**. fseval sees Feature Selection as a form of Feature Ranking, so can handle both scenarios.
3. **Interpretable AI method authors**. You wrote a new Interpretable AI method that aims to find out which features are most meaningful by ranking them. Now, the challenge is to find out how well your method ranked those features. fseval can help with this.
4. **Machine Learning practitioners**. You have a dataset and want to find out with exactly what features your models will perform best. You can use fseval to try multiple Feature Selection or Feature Ranking algorithms.



Key features 🚀:
- ...
- ...
- ...
- ...
- ...

## Getting started

Install fseval:
```shell
pip install fseval
```

Given the following [directory](https://github.com/dunnkers/fseval/tree/master/examples/quick-start) structure:
```shell
$ tree
.
├── benchmark.py
└── conf
    ├── my_config.yaml
    ├── dataset
    │   └── synthetic_dataset.yaml
    ├── ranker
    │   ├── anova.yaml
    │   └── mutual_info.yaml
    └── validator
        └── knn.yaml

4 directories, 5 files
```


And the file `benchmark.py`:
```python title="benchmark.py"
import hydra
from fseval.config import PipelineConfig
from fseval.main import run_pipeline
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


@hydra.main(config_path="conf", config_name="my_config")
def main(cfg: PipelineConfig) -> None:
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
```


We can then run a benchmark like so:
```shell
python benchmark.py --multirun ranker='glob(*)'
```
![Locale Dropdown](/img/quick-start/terminal.svg)

The data is stored in a SQLite database:

![database](/img/quick-start/database_file.png)

We can open the data using [DB Browser for SQLite](https://sqlitebrowser.org/) The experiment config is stored in the `experiments` table:
![experiments table](/img/quick-start/experiments_data.png)

We can access the validation scores in the `validation_scores` table:
![validation data](/img/quick-start/validation_data.png)

This way, we can easily compare two feature selectors: ANOVA F Value and Mutual Info ✨.