---
sidebar_label: _experiment
title: pipelines._experiment
---

## Experiment Objects

```python
@dataclass
class Experiment(AbstractEstimator)
```

#### prefit

```python
def prefit()
```

Pre-fit hook. Is executed right before calling `fit()`. Can be used to load
estimators from cache or do any other preparatory work.

#### fit

```python
def fit(X, y) -> AbstractEstimator
```

Sequentially fits all estimators in this experiment.

**Arguments**:

- `X` _np.ndarray_ - design matrix X
- `y` _np.ndarray_ - target labels y

#### postfit

```python
def postfit()
```

Post-fit hook. Is executed right after calling `fit()`. Can be used to save
estimators to cache, for example.

#### score

```python
def score(X, y, **kwargs) -> Union[Dict, pd.DataFrame, np.generic, None]
```

Sequentially scores all estimators in this experiment, and appends the scores
to a dataframe or a dict containing dataframes. Returns all accumulated scores.

