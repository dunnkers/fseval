---
sidebar_position: 0
---

# How to configure experiments
All the pipeline needs to run is a well-defined configuration. All classes are automatically instantiated, i.e. everything that has `_target_` attributes. The requirement is that whatever is passed into [`run_pipeline`](/docs/running-experiments/running-first-experiment) is a `PipelineConfig` object.

The complete pipeline configuration is as follows:

```python
@dataclass
class PipelineConfig:
    defaults: List[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"dataset": MISSING},
            {"ranker": MISSING},
            {"validator": MISSING},
            {"cv": "kfold"},
            {"storage": "local"},
            {"resample": "shuffle"},
            {"override hydra/job_logging": "colorlog"},
            {"override hydra/hydra_logging": "colorlog"},
        ]
    )
    _target_: str = "fseval.pipelines.rank_and_validate.BootstrappedRankAndValidate"
    pipeline: str = "rank-and-validate"
    dataset: DatasetConfig = MISSING
    cv: CrossValidatorConfig = MISSING
    storage: StorageConfig = MISSING
    resample: ResampleConfig = MISSING
    ranker: EstimatorConfig = MISSING
    validator: EstimatorConfig = MISSING
    callbacks: Dict[str, Any] = field(
        default_factory=lambda: {
            "_target_": "fseval.pipelines._callback_collection.CallbackCollection"
        }
    )
    n_bootstraps: int = 1
    n_jobs: Optional[int] = 1
    all_features_to_select: str = "range(1, min(50, p) + 1)"
    metrics: Dict[str, Any] = field(default_factory=dict)
```

## 
Experiments can be configured in two ways.

1. Using **YAML** files stored in a directory
2. Using **Python** (Structured Configs in Hydra)

## Using YAML files
...