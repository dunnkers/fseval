---
title: fseval.main
sidebar_position: 3
---

# fseval.main

```python
def fseval.main.run_pipeline(
    cfg: PipelineConfig,
    raise_incompatibility_errors: bool=False
) -> Optional[Dict]
```

Runs the fseval pipeline. 


**Attributes**:

| | |
|---|---|
| `cfg` : PipelineConfig | The pipeline configuration to use. |
| `raise_incompatibility_errors` : bool | Whether to raise an error when an  incompatible config was passed. Otherwise, the pipeline is exited gracefully. That is, no error is raised and the pipeline is stopped with an exit(0). |
| | |