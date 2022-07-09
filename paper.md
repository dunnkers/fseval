---
title: 'fseval: A benchmarking framework'
tags:
  - feature ranking
  - feature selection
  - benchmarking
  - machine learning
  - open-source
  - software
  - python
authors:
  - name: Jeroen G. S. Overschie
    orcid: 0000-0003-3304-3800
    affiliation: 1
  - name: Ahmad Alsahaf
    orcid: 0000-0002-0770-1390
    affiliation: 2
  - name: George Azzopardi
    orcid: 0000-0001-6552-2596
    affiliation: 1
affiliations:
 - name: Bernoulli Institute for Mathematics, Computer Science and Artificial Intelligence, University of Groningen, P.O. Box 407, 9700 AK Groningen, The Netherlands
   index: 1
 - name: Department of Biomedical Sciences of Cells and Systems, University Medical Center Groningen, University of Groningen, 9713 GZ Groningen, The Netherlands
   index: 2
date: 06 July 2022
bibliography: paper.bib
---

# Summary

The `fseval` Python package allows benchmarking Feature Selection and
Feature Ranking algorithms on a large scale, and facilitates the
comparison of multiple algorithms in a systematic way. In particular,
`fseval` enables users to run experiments in parallel and distributed
over multiple machines, and export the results to an SQL database. The
execution of an experiment can be fully determined by a configuration
file, which means the experiment results can be reproduced easily,
given only the configuration file. `fseval` has high test coverage,
continuous integration, and rich documentation. The package is open
source and can be installed through PyPI. The source code is available
at: <https://github.com/dunnkers/fseval>.

# Statement of need

Feature Selection (FS) and Feature Ranking (FR) are important and
extensively researched topics within machine learning
[@guyon_introduction_2003]. FS methods determine subsets of relevant
features in a dataset, whereas FR methods rank all features in a dataset
relative to each other in terms of their relevance. When a new FS or FR
method is developed, a benchmarking scheme is necessary to empirically
validate its effectiveness. Often, the benchmark is conducted as
follows: features are ranked by importance, then the predictive quality
of the feature subsets containing the top ranked features is evaluated
using a validation estimator. Some papers let the competing FS or FR
algorithms pick out a fixed number of top $k$ features and validate the
performance of that feature subset
[@roffo_infinite_2015; @zhao_searching_2007; @bradley_feature_1998],
whilst others evaluate multiple subsets of increasing cardinality
containing the highest ranked features
[@wojtas_feature_2020; @bennasar_feature_2015; @gu_generalized_2012; @peng_feature_2005; @kira_feature_1992; @almuallim_learning_1991].
FS algorithms that only make a binary prediction on which features to
keep, are always evaluated in the former way.

`fseval` is an open-source Python package that helps researchers
implement such benchmarks efficiently, which avoids the need for
implementing a benchmarking pipeline from scratch to test new methods.
The pipeline only requires a well-defined configuration file to run -
the rest of the pipeline is automatically executed. Because the entire
experiment setup is deterministic and captured in a configuration file,
results of any experiment can be reproduced given the configuration
file. This can be very convenient to researchers in order to prove the
integrity of their benchmarks. The scope of `fseval` is limited to
handle tabular datasets for the classification and regression
objectives.

# Key Features

`fseval` is a flexible and unbiased framework which provides as much
useful functionality as possible. Most features are optional, and can be
enabled or disabled according to what the user deems fit. The aim of the
package is to accommodate the most common benchmarking settings and
protocols that feature selection researchers use.

-   **Algorithm support**. FR or FS algorithms that estimate the
    importance of features in various ways are supported, including the
    following output attributes: (1) a
    [feature_importances\_]{.smallcaps} vector in $\mathbb{R}$, (2) a
    [ranking\_]{.smallcaps} vector in $\mathbb{Z}$ and (3) a
    [support\_]{.smallcaps} vector in $\mathbb{B}$. An estimator might
    support any combination of the output attributes. Once estimators
    are fit there is the option to save a *cached* version.

-   **Dataset** adapters. Datasets can be loaded from multiple sources
    using *adapters*. Users can implement adapters themselves by
    implementing a given interface, or use a built-in adapter class to
    load datasets from OpenML [@vanschoren_openml_2013]. Adapters might
    also be functions, which, for example, allow users to directly use
    the sklearn functions `make_classification` or `make_regression` as
    adapters to create **synthetic** datasets. Datasets might also
    define dataset feature importance *ground truths*, which can be used
    to compute metrics in the scoring stage
    (Section [3.2](#section:scoring){reference-type="ref"
    reference="section:scoring"}).

-   **Built-in integrations**. `fseval` allows exporting benchmark
    results directly to various SQL databases using SQLAlchemy
    [@bayer_sqlalchemy_2012], or to the Weights and Biases experiment
    tracker platform [@biewald_experiment_2020]. Users can create custom
    metrics and perform aggregations over the bootstrap results.

-   **Scalable and distributed** computing. Besides that the process of
    running multiple bootstraps can be distributed over the CPU,
    `fseval` also allows executing experiments on SLURM clusters
    [@yoo_slurm_2003] or on the cloud platform AWS. This is possible
    because all configuration regarding the execution of the pipeline
    can be captured in a configuration file.

-   **Reproducible** experiments. Because the entire execution state of
    the pipeline can be expressed in a single configuration file, it is
    easy to reproduce experiment results. Given that a scientist uses
    estimates that are deterministic (e.g. by fixing a
    [random_state]{.smallcaps} variable), others can reproduce the
    results, improving the scientific integrity of the work.

# The Pipeline

`fseval` executes a predefined sequence of steps, as can be seen in
Figure [1](#fig:pipeline){reference-type="ref"
reference="fig:pipeline"}.

![A schematic of the benchmarking pipeline. The input of the pipeline is
at all times a `PipelineConfig` object, processed from YAML or Python by
Hydra. After steps 1-6, steps a-d are executed for both the fitting step
and scoring step.](pipeline.svg){#fig:pipeline width="90%"}

First, in step 1, the pipeline configuration is processed using Hydra
[@yadan_hydra_2019]. Hydra is a powerful tool for creating Command Line
Interfaces in Python, allowing hierarchical representation of the
configuration. Configuration can be defined in either YAML or Python
files, or a combination of the two. The top-level config is enforced to
be of the `PipelineConfig` interface, allowing Hydra to perform
type-checking. The config is then passed to the `run_pipeline` function
in step 2. Then, after the dataset is loaded in step 3, the splits for
cross validation are determined in step 4. Each cross validation fold is
executed in a separate run of the pipeline. The training and testing
subsets are then given to the *fitting* and *scoring* steps, steps 5 and
6, respectively.

## Fitting

In the fitting step, the Feature- Ranker or Selector and validation
estimators are fit on the given training set. The validation estimator
is fit on all feature subsets that are desired to be evaluated. For
every bootstrap
$b \in \{1, \dots, \texttt{PipelineConfig.n\_bootstraps}\}$, a fit
sequence is run. The bootstraps can be distributed over CPUs by setting
`PipelineConfig.n_jobs` $> 1$. The fit process consists of the following
steps.

(a) The dataset is resampled according to the `PipelineConfig.resample`
    config, using `random_state` $= b$.

(b) The FR or FS algorithm is fit. Then, the estimator can be cached as
    a pickle file.

(c) If the ranker estimates [support\_]{.smallcaps} (*Feature
    Selection*): The selected feature subset is validated using the
    validation estimator.

(d) If the ranker estimates [feature_importances\_]{.smallcaps} or
    [ranking\_]{.smallcaps} (*Feature Ranking*) then every number in the
    list `PipelineConfig.all_features_to_select` is used to take the $k$
    best features in the ranking, and fitting the validation estimator
    on the subset.

## Scoring {#section:scoring}

After the estimators have been fit, a scoring step is executed on the
test set. By default, the validation estimator score function is
triggered and its results are stored. Depending on the estimator, this
often means *classification accuracy* for classifiers and the $R^2$
score for regressors. Besides the built-in metrics, users can install
custom metrics.

To install custom metrics, programmatic hooks are available. This
enables, for example, a user aggregate over the various validated
feature subsets and bootstrapped datasets. A user could compute the
average accuracy over all bootstraps, or compute various stability
metrics [@nogueira_stability_2018]. Another example of a custom metric
would be to compare the dataset ground-truth feature importances to the
estimated importances, which information would be available when using
*synthetic* datasets.

# Quality Assurance

To ensure the library meets the desired quality standards, the following
measures were taken.

**Test suite**. Individual modules are covered using unit tests, and
workflows are covered using integration tests. To ensure
classification-, regression-, and multi-output scenarios are all
supported, the pipeline is tested with estimators and datasets of many
types.

**Continuous Integration** (CI). On every commit, besides running the
test suite, a set of quality assurance steps are undertaken. The code is
type-checked, linted, formatted and run through a continuous code
security analysis. The CI pipeline is executed on multiple Python
versions and all major OS platforms.

**Documentation**. The docs are built using Docusaurus, allowing us to
make the documentation more interactive than what would normally be
possible with in-code documentation docstrings. The docs contain quick
start examples, images, and an API reference.

# Conclusion and Future Work

`fseval` is a comprehensive and feature rich Python library for
benchmarking Feature Ranking and Feature Selection algorithms. It allows
authors to focus on their empirical research instead of having to
implement another benchmarking pipeline - exploiting `fseval`'s support
for parallel processing, distributed computing and export possibilities.
`fseval` is open source and published on the PyPi platform. Next steps
are to include more built-in dataset adapters, metrics and export
possibilities.

# References