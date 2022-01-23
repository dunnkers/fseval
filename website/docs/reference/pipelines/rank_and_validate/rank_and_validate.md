---
sidebar_label: rank_and_validate
title: pipelines.rank_and_validate.rank_and_validate
---

## RankAndValidate Objects

```python
@dataclass
class RankAndValidate(Experiment,  RankAndValidatePipeline)
```

First fits a feature ranker (or feature selector, as long as the estimator will
attach at `feature_importances_` property to its class), and then validates the
feature ranking by fitting a &#x27;validation&#x27; estimator. The validation estimator can be
any normal sklearn estimator; just as long it supports the specified dataset type -
regression or classification.

## BootstrappedRankAndValidate Objects

```python
@dataclass
class BootstrappedRankAndValidate(Experiment,  RankAndValidatePipeline)
```

Provides an experiment that performs a &#x27;bootstrap&#x27; procedure: using different
`random_state` seeds the dataset is continuously resampled with replacement, such
that various metrics can be better approximated.

