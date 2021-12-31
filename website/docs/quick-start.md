---
sidebar_position: 1
---

# Quick start

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

