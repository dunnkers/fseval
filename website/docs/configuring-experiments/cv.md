# Cross Validation

Cross Validation is used to improve the reliability of the validation results. Built-in are `kfold` and `train_test_split`.


## K-Fold CV

`cv=kfold`

```yaml title="conf/cv/kfold.yaml"
name: K-Fold
splitter:
  _target_: sklearn.model_selection.KFold
  n_splits: 5
  shuffle: True
  random_state: 0
```

So, to use 10-fold CV, set `cv.splitter.n_splits=10`.

## Train/test split

`cv=train_test_split`

```yaml title="conf/cv/train_test_split.yaml"
name: Train/test split
splitter:
  _target_: sklearn.model_selection.ShuffleSplit
  n_splits: 1
  test_size: 0.25
  random_state: 0
```

## Custom
To make your own custom Cross Validation technique, use the following config:
```python
@dataclass
class CrossValidatorConfig:
    _target_: str = "fseval.pipeline.cv.CrossValidator"
    name: str = MISSING
    """ splitter. must be BaseCrossValidator or BaseShuffleSplit; should at least 
        implement a `split()` function. """
    splitter: Any = None
    fold: int = 0
```

For example:
```python
from hydra.core.config_store import ConfigStore

loocv = CrossValidatorConfig(
    name="Leave One Out",
    splitter={
        "_target_": "sklearn.model_selection.LeaveOneOut",
    }
)

cs = ConfigStore.instance()
cs.store(name="loocv", node=loocv, group="cv")
```

Or using YAML:
```yaml title="conf/cv/loocv.yaml"
name: Leave One Out
splitter:
    _target_: sklearn.model_selection.LeaveOneOut
```
