---
sidebar_position: 2
---

# The pipeline
fseval executes a predefined number of steps to benchmark your Feature Selector or Feature Ranker.

See the schematic illustration below:

![Pipeline main architecture](/img/the-pipeline/schematic-pipeline-main-architecture.svg)

1. First, the **config** is loaded. The config is at all times a [`PipelineConfig`](../config/PipelineConfig) object.
2. Second, a **dataset** is loaded, like defined in the [`DatasetConfig`](../config/DatasetConfig) object.
3. Third, a **cross validation** split is made. The split is done as defined in the [`CrossValidatorConfig`](../config/CrossValidatorConfig).
4. Fourth, the pipeline is **fit** using the _training_ set.
5. Lastly, the pipeline is **scored** using the _test_ set.


To get a better idea of what is happening in the pipeline, we can take a closer look:

![Pipeline zoomed](/img/the-pipeline/schematic-pipeline-zoomed.svg)

In the pipeline, the following steps are performed:
1. The data is (optionally) **resampled**. This is useful, for example, to do a [bootstrap](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)). Such, the stability of an algorithm can be determined. The resampling is configured using the [`ResampleConfig`](../config/ResampleConfig).
2. A Feature **Ranker** is fit. Any Feauture Selector or Feature Ranker is defined in the [`EstimatorConfig`](../config/EstimatorConfig).
3. According to the `all_features_to_select` parameter in the [`PipelineConfig`](../config/PipelineConfig), various feature subsets are **validated**. By default, at most 50 subsets are validated using another estimator. First, the validation estimator is fit on a subset containing only the highest ranked feature, then only the two highest ranked features, etcetera.
4. When the ranker was fit on the dataset, and the validation estimator was fit on all the feature subsets, the pipeline is **scored**. This means the ranker fitting times and the validation scores are aggregated wherever applicable, and stored into tables according to the enabled [Callbacks](../config/callbacks).