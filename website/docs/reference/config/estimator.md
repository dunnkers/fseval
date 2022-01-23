---
sidebar_label: estimator
title: config.estimator
---

## EstimatorConfig Objects

```python
@dataclass
class EstimatorConfig()
```

Configures an estimator: a Feature Ranker, Feature Selector or a validation
estimator. In the case of Feature Rankers/Selectors, set one of
`estimates_feature_importances`, `estimates_feature_support` or
`estimates_feature_ranking` to True. In the case of a validation estimator,
set `estimates_target` to True.

**Attributes**:

- `name` _str_ - Human-friendly name of this estimator.
- `estimator` _Any_ - The estimator. Must be a dictionary with a key `_target_`,
  pointing to the class that is to be instantiated. All other properties in
  the dictionary will be passed to the estimator constructor. e.g.:
  `_target_=&quot;sklearn.tree.DecisionTreeClassifier, max_depth=10)`
- `load_cache` _Optional[CacheUsage]_ - How to handle loading a cached version of the
  estimator, in a pickle file. e.g. to ignore cache, or force using it.
  To be used in combination with `PipelineConfig.storage`. See `CacheUsage`.
- `save_cache` _Optional[CacheUsage]_ - How to handle saving the fit estimator as a
  pickle file, such to facilitate caching. To be used in combination with
  `PipelineConfig.storage`. See `CacheUsage`.
- `_estimator_type` _str_ - Either &#x27;classifier&#x27;, &#x27;regressor&#x27; or &#x27;clusterer&#x27;.
- `multioutput` _bool_ - Whether this estimator supports multioutput datasets.
- `multioutput_only` _bool_ - If this estimator **only** supports multioutput
  datasets.
- `requires_positive_X` _bool_ - Whether the estimator fails if X contains negative
  values.
- `estimates_feature_importances` _bool_ - Whether the estimator estimates feature
  importances. For example, in the case of 2 features, the estimator can set
  `self.feature_importances_ = [0.9, 0.1]`, implying the estimator
  found the first feature the most useful. Alternatively, the `coef_`
  attribute can also be read and interpreted as a feature importance vector.
- `estimates_feature_support` _bool_ - Whether the estimator estimates feature
  support. A feature support vector indicates which features to include in a
  feature subset yes/no. In other words, it must be a boolean vector. It is
  to be set on the estimator `support_` attribute. Estimating the feature
  support vector is synonymous with performing feature selection. e.g.:
  `self.support_ = [True, False]`, meaning to include only the first feature
  in a feature subset.
- `estimates_feature_ranking` _bool_ - Whether the estimator ranks the features in a
  specific order. Is similar to feature importance, but does not estimate
  exact importance quantities, i.e. that are proportional to each other. An
  estimator can set the ranking using the `ranking_` attribute. e.g.:
  `self.ranking_ = [1, 0]`, such to indicate that the first feature ranks the
  highest.

