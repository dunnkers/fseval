---
sidebar_label: _ranking_validator
title: pipelines.rank_and_validate._ranking_validator
---

## RankingValidator Objects

```python
@dataclass
class RankingValidator(Experiment,  RankAndValidatePipeline)
```

Validates a feature ranking. A feature ranking is validated by comparing the
estimated feature- ranking, importance or support to the ground truth feature
importances. Generally, the ground-truth feature importances are only available
when a dataset is synthetically generated.

#### score

```python
def score(X, y, **kwargs) -> Union[Dict, pd.DataFrame, np.generic, None]
```

Scores a feature ranker, if a ground-truth on the desired dataset
feature importances is available. If this is the case, the estimated normalized
feature importances are compared to the desired ones using two metrics:
log loss and the R^2 score. Whilst the log loss converts the ground-truth
desired feature rankings to a binary value, 0/1, the R^2 score always works.

