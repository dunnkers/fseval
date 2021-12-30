---
sidebar_position: 0
---

# Quick start

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