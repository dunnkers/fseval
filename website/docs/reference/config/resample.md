---
sidebar_label: resample
title: config.resample
---

## ResampleConfig Objects

```python
@dataclass
class ResampleConfig()
```

Used to resample the dataset, before running the pipeline. Notice resampling is
performed **after** cross validation. One can, for example, use resampling **with**
replacement, such as to create multiple bootstraps of the dataset. In this way,
algorithm stability can be approximated.

**Attributes**:

- `name` _str_ - Human-friendly name of the resampling method.
- `replace` _bool_ - Whether to use resampling with replacement, yes or no.
- `sample_size` _Any_ - Can be one of two types: either a **float** from [0.0 to 1.0],
  such to select a **fraction** of the dataset to be sampled. Or,  an **int**
  from [1 to n_samples] can be used. This is the amount of exact samples to
  be selected.
- `random_state` _Optional[int]_ - Optionally, one might fix a random state to be
  used in the resampling process. In this way, results can be reproduced.
- `stratify` _Optional[List]_ - Whether to use stratified resampling. See
  sklearn.utils.resample for more information.

