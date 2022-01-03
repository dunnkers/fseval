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

Given the following directory structure:
```shell
$ tree examples/my-first-benchmark 
examples/my-first-benchmark
├── README.md
├── conf
│   ├── dataset
│   │   └── iris.yaml
│   ├── ranker
│   │   └── boruta.yaml
│   └── validator
│       └── decision_tree.yaml
└── requirements.txt

4 directories, 5 files
```


A simple example:
```python title="quickstart.py"
import hydra
from fseval.config import PipelineConfig
from fseval.main import run_pipeline

@hydra.main(config_path=None, config_name="my_config")
def main(cfg: PipelineConfig) -> None:
    run_pipeline(cfg)

if __name__ == "__main__":
    main()
```


Then running:
```shell
python quickstart.py dataset=iris ranker=boruta validator=decision_tree
```

