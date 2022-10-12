---
sidebar_position: 2
---

# The pipeline

<!--
improvements:
- consistent use of bold face; should be the action applid or the config object
-->

`fseval` executes a predefined number of steps to benchmark your Feature Selector or Feature Ranker.

See the schematic illustration below:

![Pipeline main architecture](/img/the-pipeline/pipeline.svg)

The steps (1-6) can be described as follows.

1. First, the pipeline configuration ([`PipelineConfig`](../config/PipelineConfig)) is processed using **Hydra**.

  Hydra is a powerful tool for creating Command Line Interfaces in Python, allowing hierarchical representation of the configuration. Configuration can be defined in either YAML or Python files, or a combination of the two. The top-level config is enforced to be of the `PipelineConfig` interface, allowing Hydra to perform type-checking. 

2. The config is passed to the [`run_pipeline`](../main) function. 

3. The **dataset** is loaded. Like defined in the [`DatasetConfig`](../config/DatasetConfig) object.

4. A **Cross Validation** (CV) split is made.

  The split is done as defined in the [`CrossValidatorConfig`](../config/CrossValidatorConfig). Each cross validation fold is
executed in a separate run of the pipeline.

5. The **training** subset is passed to the `fit()` step.

6. The **testing** subset is passed to the `score()` step.


## Benchmark
To get a better idea of what is happening in the pipeline, we can take a closer look at the benchmark steps of the pipeline (steps a-d).

In the pipeline, the following steps are performed:

<ol type="a">
  <li>
  
The data is (optionally) <b>resampled</b>. This is useful, for example, to do a <a href="https://en.wikipedia.org/wiki/Bootstrapping_(statistics)">bootstrap</a>. Such, the stability of an algorithm can be determined. The resampling is configured using the <a href="../config/ResampleConfig"><code>ResampleConfig</code></a>.


  
  </li>
  <li>
  
    A Feature <b>Ranker</b> is fit. Any Feature Selector or Feature Ranker is defined in the <a href="../config/EstimatorConfig"><code>EstimatorConfig</code></a>.
  
  </li>
  <li>
  
    Depending on which attributes the Feature- Ranker or Selector estimates, different validations are run.


<ol>
    <li>
    When the ranker estimates the <code>feature_importances_</code> or <code>ranking_</code> attribute, the estimated ranking is validated as follows. According to the <code>all_features_to_select</code> parameter in the <a href="../config/PipelineConfig"><code>PipelineConfig</code></a>, various feature subsets are <b>validated</b>. By default, at most 50 subsets are validated using another estimator. First, the validation estimator is fit on a subset containing only the highest ranked feature, then only the two highest ranked features, etcetera.
    
    </li>
    <li>
    In the case that a ranker estimates the <code>support_</code> attribute, that selected feature subset is validated.
    
    </li>
</ol>

  </li>
  <li>
  
    When the ranker was fit on the dataset, and the validation estimator was fit on all the feature subsets, the pipeline is <b>scored</b>. This means the ranker fitting times and the validation scores are aggregated wherever applicable, and stored into tables according to the enabled <a href="../config/callbacks">Callbacks</a>.
  
  </li>
</ol>

