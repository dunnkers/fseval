# fseval.types.Task

```python
class fseval.types.Task(Enum):
    regression = 1
    classification = 2
```

Learning task. In the case of datasets this indicates the dataset learning task, and in the case of estimators this indicates the supported estimator learning tasks.

For example, usage in a dataset config:
```yaml {2}
name: Iris Flowers
task: classification
adapter:
  _target_: fseval.adapters.openml.OpenML
  dataset_id: 61
  target_column: class
```

or:

```yaml {2}
name: Boston house prices
task: regression
adapter:
  _target_: fseval.adapters.openml.OpenML
  dataset_id: 531
  target_column: MEDV
  drop_qualitative: true
```


Used by [DatasetConfig](../../config/DatasetConfig) and [EstimatorConfig](../../config/EstimatorConfig).