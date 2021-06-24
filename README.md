# fseval

[![build status](https://github.com/dunnkers/fseval/actions/workflows/python-app.yml/badge.svg)](https://github.com/dunnkers/fseval/actions/workflows/python-app.yml) [![pypi badge](https://img.shields.io/pypi/v/fseval.svg?maxAge=3600)](https://pypi.org/project/fseval/)

A Feature Selector and Feature Ranker benchmarking library. Neatly integrates [Hydra](https://hydra.cc/) with [wandb](https://wandb.ai). Supports any [sklearn](https://scikit-learn.org/)-style feature ranker.

Main features:
- **Subset validation**. Allows you to validate the quality of a feature ranking, by running a _validation_ estimator on some of the `k` best feature subsets.
- **Online dashboard**. Experiments can be uploaded to [wandb](https://wandb.ai) for seamless experiment tracking and visualization.
- **sklearn integration**. Integrates nicely with [sklearn](https://scikit-learn.org/). Any estimator that implements `fit` is supported.
- **Reproducible configs**. Uses [Hydra](https://hydra.cc/) as a config parser, to allow configuring every part of the experiment. The full Hydra configuration is automatically uploaded to wandb.
- **Bootstrapping**. Allows you to approximate the _stability_ of an algorithm by running multiple experiments on bootstrap resampled datasets.

## Install

```shell
pip install fseval
```

## Usage
fseval is run via a CLI. Example:
```shell
fseval +dataset=synclf_easy +estimator@ranker=chi2 +estimator@validator=decision_tree
```

Which runs [Chi2](https://github.com/dunnkers/fseval/blob/master/fseval/conf/estimator/chi2.yaml) feature ranking on the [Iris](https://github.com/dunnkers/fseval/blob/master/fseval/conf/dataset/iris.yaml) dataset, and validates feature subsets using a [Decision Tree](https://github.com/dunnkers/fseval/blob/master/fseval/conf/estimator/decision_tree.yaml).

To see all the configurable options, run:
```shell
fseval --help
```


### Weights and Biases integration
Integration with [wandb](https://wandb.ai) is built-in. Create an account and login to the [CLI](https://github.com/wandb/client#-simple-integration-with-any-framework) with `wandb login`. Then, enable wandb using `callbacks="[wandb]"`:

```shell
fseval callbacks="[wandb]" +callbacks.wandb.project=fseval-readme [...]
```

This runs an experiment and uploads the results to wandb:
<p align="center">
  <img width="600" src="./docs/run-cli-example.svg">
</p>


We can now explore the results on the online dashboard:

<p align="center">
    <a href="https://wandb.ai/dunnkers/fseval-readme/runs/11b4t26e">
        <img width="650" src="./docs/run-wandb-example.png">
  </a>
</p>

✨

### Running bootstraps
_Bootstraps_ can be run, to approximate the stability of an algorithm. Bootstrapping works by creating multiple dataset permutations and running the algorithm on each of them. A simple way to create dataset permutations is to **resample with replacement**.

In fseval, bootstrapping can be configured like so:

```shell
fseval [...] resample=bootstrap n_bootstraps=8
```

To run the entire experiment 8 times, each for a resampled dataset.

In the dashboard, plots are already set up to support bootstrapping:
<p align="center">
  <img width="600" src="./docs/run-bootstraps-example.svg">
</p>

Shows the validation results for **25** bootstraps. ✨

### Multiprocessing
The experiment can run in parallel. The list of bootstraps is distributed over the CPU's. To use all available processors:

```shell
fseval [...] n_jobs=-1
```

Alternatively, set `n_jobs` to the specific amount of processors to use. e.g. `n_jobs=4` if you have a quad-core.

When using bootstraps, it can be efficient to use an amount that is divisible by the amount of CPU's:

```shell
fseval [...] resample=bootstrap n_bootstraps=8 n_jobs=4
```

would cause all 8 CPU's to be utilized efficiently. 

### Distributed processing
Since fseval uses Hydra, all its plugins can also be used. Some plugins for distributed processing are:

- [RQ launcher](https://hydra.cc/docs/plugins/rq_launcher/). Uses Redis Queue ([RQ](https://python-rq.org/)) to launch jobs.
- [Submitit Launcher](https://hydra.cc/docs/plugins/submitit_launcher). Submits jobs directly to a [SLURM](https://slurm.schedmd.com/) cluster. See the [example setup](https://github.com/dunnkers/fseval/tree/master/examples/slurm-hpc-benchmark).

Example:
```shell
fseval --multirun [...] hydra/launcher=rq
```

To submit jobs to RQ.

### Configuring a Feature Ranker
The entirety of the config can be overriden like pleased. Like such, also feature rankers can be configured. For example:

```shell
fseval [...] +validator.classifier.estimator.criterion=entropy
```

Changes the Decision Tree criterion to entropy.

## Config directory
Any configuration can also be loaded from a dir. It is configured like so:

```shell
fseval --config-dir ./conf
```

With the `./conf` directory containing:

```shell
.
└── conf
    ├── estimator
    │   └── my_custom_ranker.yaml
    └── dataset
        └── my_custom_dataset.yaml
```

We can now use the newly installed estimator and dataset:

```shell
fseval --config-dir ./conf +estimator@ranker=my_custom_ranker +dataset=my_custom_dataset
```

🙌🏻


Where `my_custom_ranker.yaml` would be any [estimator](https://github.com/dunnkers/fseval/tree/master/fseval/conf/estimator) definition, and `my_custom_dataset.yaml` any [dataset](https://github.com/dunnkers/fseval/tree/master/fseval/conf/dataset) dataset definition.


## Built-in Feature Rankers
A number of rankers are already built-in, which can be used without further configuring. See:

| Ranker | Dependency | Command line argument
--- | --- | ---
[ANOVA F-Value](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.f_classif) | - | `+estimator@ranker=anova_f_value`
[Boruta](https://github.com/scikit-learn-contrib/boruta_py) | `pip install Boruta` | `+estimator@ranker=boruta`
[Chi2](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html) | - | `+estimator@ranker=chi2`
[Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) | - | `+estimator@ranker=decision_tree`
[FeatBoost](https://github.com/amjams/FeatBoost) | `pip install git+https://github.com/amjams/FeatBoost.git` | `+estimator@ranker=featboost`
[MultiSURF](https://github.com/EpistasisLab/scikit-rebate) | `pip install skrebate` | `+estimator@ranker=multisurf`
[Mutual Info](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html) | - | `+estimator@ranker=mutual_info`
[ReliefF](https://github.com/EpistasisLab/scikit-rebate) | `pip install skrebate` | `+estimator@ranker=relieff`
[Stability Selection](https://github.com/scikit-learn-contrib/stability-selection) | `pip install git+https://github.com/dunnkers/stability-selection.git matplotlib` ℹ️ | `+estimator@ranker=stability_selection`
[TabNet](https://github.com/dreamquark-ai/tabnet) | `pip install pytorch-tabnet` | `+estimator@ranker=tabnet`
[XGBoost](https://xgboost.readthedocs.io/) | `pip install xgboost` | `+estimator@ranker=xgb`
[Infinite Selection](https://github.com/giorgioroffo/Infinite-Feature-Selection) | `pip install git+https://github.com/dunnkers/infinite-selection.git` ℹ️ | `+estimator@ranker=infinite_selection`


ℹ️ This library was customized to make it compatible with the fseval pipeline.

### About
Built by [Jeroen Overschie](https://dunnkers.com/) as part of a Masters Thesis.
→ (Data Science and Computational Complexity, University of Groningen)
