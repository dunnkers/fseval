---
sidebar_label: _subset_validator
title: pipelines.rank_and_validate._subset_validator
---

## SubsetValidator Objects

```python
@dataclass
class SubsetValidator(Experiment,  RankAndValidatePipeline)
```

Validates one feature subset using a given validation estimator. i.e. it first
performs feature selection using the ranking made available in the fitted ranker,
`self.ranker`, and then fits/scores an estimator on that subset.

#### score

```python
def score(X, y, **kwargs) -> Union[Dict, pd.DataFrame, np.generic, None]
```

Compute validator score. Uses the `score()` function configured in the
validator itself. For example, k-NN has a `score()` function that uses the
`accuracy` score. To customize, override the `score()` function in the
validation estimator.

