---
sidebar_label: pipeline
title: config.pipeline
---

## PipelineConfig Objects

```python
@dataclass
class PipelineConfig()
```

The complete configuration needed to run the fseval pipeline.

**Attributes**:

- `dataset` _DatasetConfig_ - Determines the dataset to use for this experiment.
- `ranker` _EstimatorConfig_ - A Feature Ranker or Feature Selector.
- `validator` _EstimatorConfig_ - Some estimator to validate the feature subsets.
- `cv` _CrossValidatorConfig_ - The CV method and split to use in this experiment.
- `resample` _ResampleConfig_ - Dataset resampling; e.g. with or without replacement.
- `storage` _StorageConfig_ - A storage method used to store the fit estimators.
- `callbacks` _Dict[str, Any]_ - Callbacks. Provide hooks for storing the config or
  results.
- `metrics` _Dict[str, Any]_ - Metrics allow custom computation after any pipeline
  stage.
- `n_bootstraps` _int_ - Amount of &#x27;bootstraps&#x27; to run. A bootstrap means running
  the pipeline again but with a resampled (see `resample`) version of the
  dataset. This allows estimating stability, for example.
- `n_jobs` _Optional[int]_ - Amount of CPU&#x27;s to use for computing each bootstrap.
  This thus distributes the amount of bootstraps over CPU&#x27;s.
- `all_features_to_select` _str_ - Once the ranker has been fit, this determines the
  feature subsets to validate. By default, at most 50 subsets containing the
  highest ranked features are validated.

