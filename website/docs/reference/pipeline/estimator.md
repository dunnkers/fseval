---
sidebar_label: estimator
title: pipeline.estimator
---

## Estimator Objects

```python
@dataclass
class Estimator(AbstractEstimator,  EstimatorConfig)
```

#### \_\_post\_init\_\_

```python
def __post_init__()
```

Perform compatibility checks after initialization. Check whether this
estimator is compatible with this dataset. Otherwise, raise an error. This error
can be caught so the pipeline can be exited gracefully.

#### fit\_time\_

```python
@property
def fit_time_()
```

&quot;Retrieves the estimator fitting time. Either the cached fitting time, or
the time that was just recorded.

#### fit\_time\_

```python
@fit_time_.setter
def fit_time_(fit_time_)
```

Store the recorded fitting time. Stored in an attribute on the estimator
itself, so the fitting time is cached inside the estimator object.

