---
sidebar_label: types
title: types
---

## Task Objects

```python
class Task(Enum)
```

Learning task. In the case of datasets this indicates the dataset learning task,
and in the case of estimators this indicates the supported estimator learning tasks.

## CacheUsage Objects

```python
class CacheUsage(Enum)
```

Determines how cache usage is handled. In the case of **loading** caches:

- `allow`: program might use cache; if found and could be restored
- `must`: program should fail if no cache found
- `never`: program should not load cache even if found

When **saving** caches:
- `allow`: program might save cache; no fatal error thrown when fails
- `must`: program must save cache; throws error if fails (e.g. due to out of memory)
- `never`: program does not try to save a cached version

## AbstractMetric Objects

```python
class AbstractMetric()
```

#### score\_bootstrap

```python
def score_bootstrap(ranker: AbstractEstimator, validator: AbstractEstimator, callbacks: Callback, scores: Dict, **kwargs, ,) -> Dict
```

Aggregated metrics for the bootstrapped pipeline.

#### score\_pipeline

```python
def score_pipeline(scores: Dict, callbacks: Callback, **kwargs) -> Dict
```

Aggregated metrics for the pipeline.

#### score\_ranking

```python
def score_ranking(scores: Union[Dict, pd.DataFrame], ranker: AbstractEstimator, bootstrap_state: int, callbacks: Callback, feature_importances: Optional[np.ndarray] = None) -> Union[Dict, pd.DataFrame]
```

Metrics for validating a feature ranking, e.g. using a ground-truth.

#### score\_support

```python
def score_support(scores: Union[Dict, pd.DataFrame], validator: AbstractEstimator, X, y, callbacks: Callback, **kwargs, ,) -> Union[Dict, pd.DataFrame]
```

Metrics for validating a feature support vector. e.g., this is an array
indicating yes/no which features to include in a feature subset. The array is
validated by running the validation estimator on this feature subset.

#### score\_dataset

```python
def score_dataset(scores: Union[Dict, pd.DataFrame], callbacks: Callback, **kwargs) -> Union[Dict, pd.DataFrame]
```

Aggregated metrics for all feature subsets. e.g. 50 feature subsets for
p &gt;= 50.

#### score\_subset

```python
def score_subset(scores: Union[Dict, pd.DataFrame], validator: AbstractEstimator, X, y, callbacks: Callback, **kwargs, ,) -> Union[Dict, pd.DataFrame]
```

Metrics for validation estimator. Validates 1 feature subset.

