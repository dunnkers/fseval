---
sidebar_label: dataset
title: config.dataset
---

## DatasetConfig Objects

```python
@dataclass
class DatasetConfig()
```

Configures a dataset, to be used in the pipeline. Can be loaded from various sources
using an &#x27;adapter&#x27;.

**Attributes**:

- `name` _str_ - Human-readable name of dataset.
- `task` _Task_ - Either Task.classification or Task.regression.
- `adapter` - Dataset adapter. must be of fseval.types.AbstractAdapter type,
  i.e. must implement a get_data() -&gt; (X, y) method. Can also be a callable;
  then the callable must return a tuple (X, y).
- `adapter_callable` - Adapter class callable. the function to be called on the
  instantiated class to fetch the data (X, y). is ignored when the target
  itself is a function callable.
- `feature_importances` _Optional[Dict[str, float]]_ - Weightings indicating relevant
  features or instances. should be a dict with each key and value like the
  following pattern:
  X[&lt;numpy selector&gt;] = &lt;float&gt;

**Example**:

  X[:, 0:3] = 1.0
  which sets the 0-3 features as maximally relevant and all others
  minimally relevant.
- `group` _Optional[str]_ - An optional group attribute, such to group datasets in
  the analytics stage.
- `domain` _Optional[str]_ - Dataset domain, e.g. medicine, finance, etc.

