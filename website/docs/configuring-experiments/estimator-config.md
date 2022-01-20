---
sidebar_position: 2
---

# EstimatorConfig
Both feature rankers and validators are defined using the `EstimatorConfig`. The config for both is like below:

```python
@dataclass
class EstimatorConfig:
    """
    Configures an estimator: a Feature Ranker, Feature Selector or a validation 
    estimator. In the case of Feature Rankers/Selectors, set one of 
    `estimates_feature_importances`, `estimates_feature_support` or  
    `estimates_feature_ranking` to True. In the case of a validation estimator,
    set `estimates_target` to True.

    Attributes:
        name (str): Human-friendly name of this estimator.
        estimator (Any): The estimator. Must be a dictionary with a key `_target_`, 
            pointing to the class that is to be instantiated. All other properties in 
            the dictionary will be passed to the estimator constructor. e.g.:
            `_target_="sklearn.tree.DecisionTreeClassifier, max_depth=10)`
        load_cache (Optional[CacheUsage]): How to handle loading a cached version of the
            estimator, in a pickle file. e.g. to ignore cache, or force using it. 
            To be used in combination with `PipelineConfig.storage`. See `CacheUsage`.
        save_cache (Optional[CacheUsage]): How to handle saving the fit estimator as a 
            pickle file, such to facilitate caching. To be used in combination with 
            `PipelineConfig.storage`. See `CacheUsage`.
        _estimator_type (str): Either 'classifier', 'regressor' or 'clusterer'.
        multioutput (bool): Whether this estimator supports multioutput datasets.
        multioutput_only (bool): If this estimator **only** supports multioutput 
            datasets.
        requires_positive_X (bool): Whether the estimator fails if X contains negative 
            values.
        estimates_feature_importances (bool): Whether the estimator estimates feature
            importances. For example, in the case of 2 features, the estimator can set 
            `self.feature_importances_ = [0.9, 0.1]`, implying the estimator
            found the first feature the most useful. Alternatively, the `coef_` 
            attribute can also be read and interpreted as a feature importance vector.
        estimates_feature_support (bool): Whether the estimator estimates feature 
            support. A feature support vector indicates which features to include in a 
            feature subset yes/no. In other words, it must be a boolean vector. It is 
            to be set on the estimator `support_` attribute. Estimating the feature 
            support vector is synonymous with performing feature selection. e.g.:
            `self.support_ = [True, False]`, meaning to include only the first feature
            in a feature subset.
        estimates_feature_ranking (bool): Whether the estimator ranks the features in a
            specific order. Is similar to feature importance, but does not estimate 
            exact importance quantities, i.e. that are proportional to each other. An
            estimator can set the ranking using the `ranking_` attribute. e.g.:
            `self.ranking_ = [1, 0]`, such to indicate that the first feature ranks the
            highest.
    """
    name: str = MISSING
    estimator: Any = None
    load_cache: Optional[CacheUsage] = CacheUsage.allow
    save_cache: Optional[CacheUsage] = CacheUsage.allow
    # tags
    _estimator_type: str = MISSING  # 'classifier', 'regressor' or 'clusterer'
    multioutput: bool = False
    multioutput_only: bool = False
    requires_positive_X: bool = False
    estimates_feature_importances: bool = False
    estimates_feature_support: bool = False
    estimates_feature_ranking: bool = False
    estimates_target: bool = False
    # runtime properties. do not override these.
    task: Task = II("dataset.task")
    is_multioutput_dataset: bool = II("dataset.multioutput")

    # required for instantiation
    _target_: str = "fseval.pipeline.estimator.Estimator"
```

Example Feature Ranker:
```yaml title="conf/ranker/relieff.yaml"
name: ReliefF
estimator:
  _target_: skrebate.ReliefF
_estimator_type: classifier
estimates_feature_importances: true
```

Example Feature Ranker:
```yaml title="conf/ranker/boruta.yaml"
name: Boruta
estimator:
  _target_: boruta.boruta_py.BorutaPy
  estimator:
    _target_: sklearn.ensemble.RandomForestClassifier
  n_estimators: auto
_estimator_type: classifier
multioutput: false
estimates_feature_importances: false
estimates_feature_support: true
estimates_feature_ranking: true
```

Example validator:
```yaml title="conf/validator/knn.yaml"
name: k-NN
estimator:
  _target_: sklearn.neighbors.KNeighborsClassifier
_estimator_type: classifier
multioutput: false
estimates_target: true
```

→ For more examples see [this](https://github.com/dunnkers/fseval/tree/master/tests/integration/conf/) directory.

<details>
<summary>
`Task` Enum
</summary>
For setting the `task` attribute:

```python 
class Task(Enum):
    """Learning task. In the case of datasets this indicates the dataset learning task,
    and in the case of estimators this indicates the supported estimator learning tasks.
    """

    regression = 1
    classification = 2
```
</details>

<details>
<summary>
`CacheUsage` Enum
</summary>

For setting the `load_cache` and `save_cache` attributes:
```python

class CacheUsage(Enum):
    """
    Determines how cache usage is handled. In the case of **loading** caches:

    - `allow`: program might use cache; if found and could be restored
    - `must`: program should fail if no cache found
    - `never`: program should not load cache even if found

    When **saving** caches:
    - `allow`: program might save cache; no fatal error thrown when fails
    - `must`: program must save cache; throws error if fails (e.g. due to out of memory)
    - `never`: program does not try to save a cached version
    """

    allow = 1
    must = 2
    never = 3
```
</details>
