<p align="center">
  <img width="100%" src="./docs/header.png">
</p>

# fseval

[![build status](https://github.com/dunnkers/fseval/actions/workflows/python-app.yml/badge.svg)](https://github.com/dunnkers/fseval/actions/workflows/python-app.yml) [![pypi badge](https://img.shields.io/pypi/v/fseval.svg?maxAge=3600)](https://pypi.org/project/fseval/) [![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Downloads](https://pepy.tech/badge/fseval/month)](https://pepy.tech/project/fseval) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fseval) [![codecov](https://codecov.io/gh/dunnkers/fseval/branch/master/graph/badge.svg?token=R5ZXH8UPCI)](https://codecov.io/gh/dunnkers/fseval) [![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/dunnkers/fseval.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/dunnkers/fseval/context:python)

A Feature Ranker benchmarking library. Useful for Feature Selection and Interpretable AI methods. Allows plotting feature importance scores on an online dashboard. Neatly integrates [Hydra](https://hydra.cc/) with [wandb](https://wandb.ai).

Any [sklearn](https://scikit-learn.org/) style estimator can be used as a Feature Ranker. Estimator must estimate at least one of:

1. **Feature importance**, using `feature_importances_`.
2. **Feature subset**, using `feature_support_`.
3. **Feature ranking**, using `feature_ranking_`.

Main functionality:
- 📊 **Online dashboard**. Experiments can be uploaded to [wandb](https://wandb.ai) for seamless experiment tracking and visualization. Feature importance and subset validation plots are built-in. 
- 🔄 **Scikit-Learn integration**. Integrates nicely with [sklearn](https://scikit-learn.org/). Any estimator that implements `fit` is supported.
- 🗄 **Dataset adapters**. Datasets can be loaded dynamically using an _adapter_. [OpenML](https://www.openml.org/search?type=data) support is built-in.
- 🎛 **Synthetic dataset generation**. Synthetic datasets can be generated and configured right in the library itself.
- 📌 **Relevant features ground-truth**. Datasets can have ground-truth relevant features defined, so the estimated versus the ground-truth feature importance is automatically plotted in the dashboard.
- ⚜️ **Subset validation**. Allows you to validate the quality of a feature ranking, by running a _validation_ estimator on some of the `k` best feature subsets.
- ⚖️ **Bootstrapping**. Allows you to approximate the _stability_ of an algorithm by running multiple experiments on bootstrap resampled datasets.
- ⚙️ **Reproducible configs**. Uses [Hydra](https://hydra.cc/) as a config parser, to allow configuring every part of the experiment. The config can be uploaded to wandb, so the experiment can be replayed later.

## Install

```shell
pip install fseval
```

## Usage
fseval is run via a CLI. Example:
```shell
fseval \
  +dataset=synclf_easy \
  +estimator@ranker=chi2 \
  +estimator@validator=decision_tree
```

Which runs [Chi2](https://github.com/dunnkers/fseval/blob/master/fseval/conf/estimator/chi2.yaml) feature ranking on the [Synclf easy](https://github.com/dunnkers/fseval/blob/master/fseval/conf/dataset/synclf_easy.yaml) dataset, and validates feature subsets using a [Decision Tree](https://github.com/dunnkers/fseval/blob/master/fseval/conf/estimator/decision_tree.yaml).

<p align="center">
  <img width="600" src="./docs/run-cli-example.svg">
</p>

To see all the configurable options, run:
```shell
fseval --help
```

### SQL Alchemy
Data can be exported to [SQLAlchemy](https://www.sqlalchemy.org/) supported databases. That is: SQLite, Postgresql, MySQL, Oracle and [others](https://docs.sqlalchemy.org/en/14/dialects/). Install the package first.

```shell
pip install SQLAlchemy
```

To export data using SQLAlchemy, use:

```shell
fseval \
  ... \
  callbacks="[sql_alchemy]" \
  +callbacks.wandb.project=fseval-readme \
  ++callbacks.sql_alchemy.engine.url=<your_dbapi_connection_url>
```

See the [docs](https://docs.sqlalchemy.org/en/14/core/engines.html#sqlalchemy.create_engine) on construction a DBAPI connection URL.

### Weights and Biases integration
Integration with [wandb](https://wandb.ai) is built-in. First, install the package:

```shell
pip install wandb
```

Create an account and login to the [CLI](https://github.com/wandb/client#-simple-integration-with-any-framework) with `wandb login`. Then, enable wandb using `callbacks="[wandb]"`:

```shell
fseval \
  callbacks="[wandb]" \
  +callbacks.wandb.project=fseval-readme \
  +dataset=synclf_easy \
  +estimator@ranker=chi2 \
  +estimator@validator=decision_tree
```

We can now explore the results on the online dashboard:

<p align="center">
    <a href="https://wandb.ai/dunnkers/fseval-readme/runs/11b4t26e">
        <img width="650" src="./docs/run-wandb-example.png">
  </a>
</p>

✨

### Running bootstraps
_Bootstraps_ can be run, to approximate the stability of an algorithm. Bootstrapping works by creating multiple dataset permutations and running the algorithm on each of them. A simple way to create dataset permutations is to **resample with replacement**.

In fseval, bootstrapping can be configured with `resample=bootstrap`:

```shell
fseval \
  resample=bootstrap \
  n_bootstraps=8 \
  +dataset=synclf_easy \
  +estimator@ranker=chi2 \
  +estimator@validator=decision_tree 
```

To run the entire experiment 8 times, each for a resampled dataset.

In the dashboard, plots are already set up to support bootstrapping:
<p align="center">
  <img width="600" src="./docs/run-bootstraps-example.svg">
</p>

Shows the validation results for **25** bootstraps. ✨

### Launching multiple experiments at once
To launch multiple experiments, use [`--multirun`](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run).

```shell
fseval \
  --multirun \
  +dataset=synclf_easy \
  +estimator@ranker=[boruta,featboost,chi2] \
  +estimator@validator=decision_tree 
```

Which launches 3 jobs.

See the multirun overriding [syntax](https://hydra.cc/docs/advanced/override_grammar/extended). For example, you can select multiple groups using `[]`, a range using `range(start, stop, step)` and all options using `glob(*)`.

### Multiprocessing
The experiment can run in parallel. The list of bootstraps is distributed over the CPU's. To use all available processors set `n_jobs=-1`:

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

Changes the Decision Tree criterion to entropy. One could perform a **hyper-parameter sweep** over some parameters like so:

```shell
fseval --multirun [...] +validator.classifier.estimator.criterion=entropy,gini
```

or, in case of a ranker:

```shell
fseval --multirun [...] +ranker.classifier.estimator.learning_rate="range(0.1, 2.1, 0.1)"
```

Which launches 20 jobs with different learning rates (this hyper-parameter applies to `+estimator@ranker=featboost`). See [multirun](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run) docs for syntax.

## Config directory
The amount of command-line arguments quickly adds up. Any configuration can also be loaded from a dir. It is configured with [`--config-dir`](https://hydra.cc/docs/advanced/hydra-command-line-flags):

```shell
fseval --config-dir ./conf
```

With the `./conf` directory containing:
```shell
.
└── conf
    └── experiment
        └── my_experiment_presets.yaml
```

Then, `my_experiment_presets.yaml` can contain:

```yaml
# @package _global_
defaults:
  - override /resample: bootstrap
  - override /callbacks:
    - wandb

callbacks:
  wandb:
    project: my-first-benchmark

n_bootstraps: 20
n_jobs: 4
```

Which configures wandb, bootstrapping, and multiprocessing. ✓ See the [example config](https://github.com/dunnkers/fseval/tree/master/examples/my-first-benchmark).

Also, extra estimators or datasets can be added:
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

## About
Built by at the University of Groningen.

---

<p align="center">2021 — <a href="https://dunnkers.com/">Jeroen Overschie</a></p>