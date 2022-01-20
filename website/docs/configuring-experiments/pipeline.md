---
sidebar_position: 0
---

# PipelineConfig
All the pipeline needs to run is a well-defined configuration. The requirement is that whatever is passed into [`run_pipeline`](/docs/running-experiments/running-first-experiment) is a `PipelineConfig` object.

The complete pipeline configuration is as follows:

```python
@dataclass
class PipelineConfig:
    """
    The complete configuration needed to run the fseval pipeline.

    Attributes:
        dataset (DatasetConfig): Determines the dataset to use for this experiment.
        ranker (EstimatorConfig): A Feature Ranker or Feature Selector.
        validator (EstimatorConfig): Some estimator to validate the feature subsets.
        cv (CrossValidatorConfig): The CV method and split to use in this experiment.
        resample (ResampleConfig): Dataset resampling; e.g. with or without replacement.
        storage (StorageConfig): A storage method used to store the fit estimators.
        callbacks (Dict[str, Any]): Callbacks. Provide hooks for storing the config or results.
        metrics (Dict[str, Any]): Metrics allow custom computation after any pipeline stage.
        n_bootstraps (int): Amount of 'bootstraps' to run. A bootstrap means running the pipeline
            again but with a resampled (see `resample`) version of the dataset. This allows estimating
            stability, for example.
        n_jobs (Optional[int]): Amount of CPU's to use for computing each bootstrap. This thus
            distributes the amount of bootstraps over CPU's.
        all_features_to_select (str): Once the ranker has been fit, this determines the feature
            subsets to validate. By default, at most 50 subsets containing the highest ranked
            features are validated.
    """

    dataset: DatasetConfig = MISSING
    ranker: EstimatorConfig = MISSING
    validator: EstimatorConfig = MISSING
    cv: CrossValidatorConfig = MISSING
    resample: ResampleConfig = MISSING
    storage: StorageConfig = MISSING
    callbacks: Dict[str, Any] = field(default_factory=lambda: {})
    metrics: Dict[str, Any] = field(default_factory=lambda: {})
    n_bootstraps: int = 1
    n_jobs: Optional[int] = 1
    all_features_to_select: str = "range(1, min(50, p) + 1)"

    # default values for the above.
    defaults: List[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"dataset": MISSING},
            {"ranker": MISSING},
            {"validator": MISSING},
            {"cv": "kfold"},
            {"storage": "local"},
            {"resample": "shuffle"},
            {"callbacks": ["to_sql"]},
            {"metrics": ["feature_importances", "ranking_scores", "validation_scores"]},
            {"override hydra/job_logging": "colorlog"},
            {"override hydra/hydra_logging": "colorlog"},
        ]
    )

    # required for instantiation
    _target_: str = "fseval.pipelines.rank_and_validate.BootstrappedRankAndValidate"
```

## 
Experiments can be configured in two ways.

1. Using **YAML** files stored in a directory
2. Using **Python** (Structured Configs in Hydra)

## Using YAML files
...