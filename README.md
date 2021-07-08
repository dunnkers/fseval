<p align="center">
  <img width="100%" src="./docs/header.png">
</p>

# fseval

[![build status](https://github.com/dunnkers/fseval/actions/workflows/python-app.yml/badge.svg)](https://github.com/dunnkers/fseval/actions/workflows/python-app.yml) [![pypi badge](https://img.shields.io/pypi/v/fseval.svg?maxAge=3600)](https://pypi.org/project/fseval/)

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


### Weights and Biases integration
Integration with [wandb](https://wandb.ai) is built-in. Create an account and login to the [CLI](https://github.com/wandb/client#-simple-integration-with-any-framework) with `wandb login`. Then, enable wandb using `callbacks="[wandb]"`:

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

## Built-ins
<details>
<summary>Built-in Feature Rankers</summary>

A number of rankers are already built-in, which can be used without further configuring. See:

| Method | CLI | Classif- ication | Regr- ession | Multi output | Feature importance | Feature support | Feature ranking |
|-|-|-|-|-|-|-|-|
| ANOVA F-value ([sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html)) | `anova_f_value` <details><summary>command</summary>`+estimator@ranker=anova_f_value`</details>| ✓ | ✓ |  | ✓ |  |  |
| Boruta ([github](https://github.com/scikit-learn-contrib/boruta_py)) <details><summary>install</summary>`pip install Boruta`</details> | `boruta` <details><summary>command</summary>`+estimator@ranker=boruta`</details>| ✓ |  |  |  | ✓ | ✓ |
| Chi-Squared ([sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html)) | `chi2` <details><summary>command</summary>`+estimator@ranker=chi2`</details>| ✓ |  |  | ✓ |  |  |
| Decision Tree ([sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)) | `decision_tree` <details><summary>command</summary>`+estimator@ranker=decision_tree`</details>| ✓ | ✓ | ✓ | ✓ |  |  |
| FeatBoost ([github](https://github.com/amjams/FeatBoost)) <details><summary>install</summary>`pip install git+https://github.com/amjams/FeatBoost.git`</details> | `featboost` <details><summary>command</summary>`+estimator@ranker=featboost`</details>| ✓ |  |  | ✓ | ✓ |  |
| Infinite Selection ([github](https://github.com/giorgioroffo/Infinite-Feature-Selection)) <details><summary>install</summary>`pip install git+https://github.com/dunnkers/infinite-selection.git` ℹ️</details> | `infinite_selection` <details><summary>command</summary>`+estimator@ranker=infinite_selection`</details>| ✓ |  |  | ✓ |  | ✓ |
| MultiSURF ([github](https://github.com/EpistasisLab/scikit-rebate)) <details><summary>install</summary>`pip install skrebate`</details> | `multisurf` <details><summary>command</summary>`+estimator@ranker=multisurf`</details>| ✓ | ✓ |  | ✓ |  |  |
| Mutual Info ([github](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html)) | `mutual_info` <details><summary>command</summary>`+estimator@ranker=mutual_info`</details>| ✓ | ✓ |  | ✓ |  |  |
| ReliefF ([github](https://github.com/EpistasisLab/scikit-rebate)) <details><summary>install</summary>`pip install skrebate`</details> | `relieff` <details><summary>command</summary>`+estimator@ranker=relieff`</details>| ✓ | ✓ |  | ✓ |  |  |
| Stability Selection ([github](https://github.com/scikit-learn-contrib/stability-selection)) <details><summary>install</summary>`pip install git+https://github.com/dunnkers/stability-selection.git matplotlib` ℹ️</details> | `stability_selection` <details><summary>command</summary>`+estimator@ranker=stability_selection`</details>| ✓ |  |  | ✓ | ✓ |  |
| TabNet ([github](https://github.com/dreamquark-ai/tabnet)) <details><summary>install</summary>`pip install pytorch-tabnet`</details> | `tabnet` <details><summary>command</summary>`+estimator@ranker=tabnet`</details>| ✓ | ✓ | ✓ | ✓ |  |  |
| XGBoost ([github](https://xgboost.readthedocs.io/)) <details><summary>install</summary>`pip install xgboost`</details> | `xgb` <details><summary>command</summary>`+estimator@ranker=xgb`</details>| ✓ | ✓ |  | ✓ |  |  |

ℹ️ This library was customized to make it compatible with the fseval pipeline.

If you would like to install simply all dependencies, download the fseval [requirements.txt](https://github.com/dunnkers/fseval/blob/master/requirements.txt) file and run `pip install -r requirements.txt`.

</details>

<details>
<summary>Built-in datasets</summary>

| Name dataset                      | CLI | n | p | Task   | Multi- output | Domain                       | Group                          |
|-------------------------------------------|-------|-------------|-------------|----------------|------------------------|--------------------------------------|----------------------------------------|
| Boston house prices      | `boston` <details><summary>command</summary>`+dataset=boston`</details> | 506         | 11          | Regression     | No                     | Finance                              | -                                      |
| Additive                                  | `chen_additive` <details><summary>command</summary>`+dataset=chen_additive`</details> | 10000       | 10          | Regression     | Yes                    | Synthetic                            | Chen et al. |
| Orange                                    | `chen_orange` <details><summary>command</summary>`+dataset=chen_orange`</details> | 10000       | 10          | Regression     | Yes                    | Synthetic                            | Chen et al. |
| XOR                                       | `chen_xor` <details><summary>command</summary>`+dataset=chen_xor`</details> | 10000       | 10          | Regression     | Yes                    | Synthetic                            | Chen et al. |
| Climate Model Simulation | 540         | `climate_model_simulation` <details><summary>command</summary>`+dataset=climate_model_simulation`</details> | 18          | Classification | No                     | Nature                               | OpenML-CC18 |
| Cylinder bands                            | `cylinder_bands` <details><summary>command</summary>`+dataset=cylinder_bands`</details> | 5456        | 24          | Classification | No                     | Mechanics                            | OpenML-CC18 |
| Iris Flowers                              | `iris` <details><summary>command</summary>`+dataset=iris`</details> | 150         | 4           | Classification | No                     | Nature                               | -                                      |
| Madelon                                   | `madelon` <details><summary>command</summary>`+dataset=madelon`</details> | 2600        | 500         | Classification | No                     | Synthetic                            | Guyon |
| Multifeat Pixel                           | `mfeat_pixel` <details><summary>command</summary>`+dataset=mfeat_pixel`</details> | 2000        | 240         | Classification | No                     | OCR                                  | OpenML-CC18 |
| Nomao                                     | `nomao` <details><summary>command</summary>`+dataset=nomao`</details> | 34465       | 89          | Classification | No                     | Geodata                              | OpenML-CC18 |
| Ozone Levels                              | `ozone_levels` <details><summary>command</summary>`+dataset=ozone_levels`</details> | 2534        | 72          | Classification | No                     | Nature                               | OpenML-CC18 |
| Phoneme                                   | `phoneme` <details><summary>command</summary>`+dataset=phoneme`</details> | 5404        | 5           | Classification | No                     | Biomedical                           | OpenML-CC18 |
| Synclf easy                               | `synclf_easy` <details><summary>command</summary>`+dataset=synclf_easy`</details> | 10000       | 20          | Classification | No                     | Synthetic                            | Synclf                                 |
| Synclf medium                             | `synclf_medium` <details><summary>command</summary>`+dataset=synclf_medium`</details> | 10000       | 30          | Classification | No                     | Synthetic                            | Synclf                                 |
| Synclf hard                               | `synclf_hard` <details><summary>command</summary>`+dataset=synclf_hard`</details> | 10000       | 50          | Classification | No                     | Synthetic                            | Synclf                                 |
| Synclf very hard                          | `synclf_very_hard` <details><summary>command</summary>`+dataset=synclf_very_hard`</details> | 10000       | 50          | Classification | No                     | Synthetic                            | Synclf                                 |
| Synreg easy                               | `synreg_easy` <details><summary>command</summary>`+dataset=synreg_easy`</details> | 10000       | 10          | Regression     | No                     | Synthetic                            | Synreg                                 |
| Synreg medium                             | `synreg_medium` <details><summary>command</summary>`+dataset=synreg_medium`</details> | 10000       | 10          | Regression     | No                     | Synthetic                            | Synreg                                 |
| Synreg hard                               | `synreg_hard` <details><summary>command</summary>`+dataset=synreg_hard`</details> | 10000       | 20          | Regression     | No                     | Synthetic                            | Synreg                                 |
| Synreg hard                               | `synreg_very_hard` <details><summary>command</summary>`+dataset=synreg_very_hard`</details> | 10000       | 20          | Regression     | No                     | Synthetic                            | Synreg                                 |
| Texture                                   | `texture` <details><summary>command</summary>`+dataset=texture`</details> | 5500        | 40          | Classification | No                     | Pattern Recognition | OpenML-CC18 |
| Wall Robot Navigation    | `wall_robot_navigation` <details><summary>command</summary>`+dataset=wall_robot_navigation`</details> | 5456        | 24          | Classification | No                     | Mechanics                            | OpenML-CC18 |
</details>

## About
Built by [Jeroen Overschie](https://dunnkers.com/) as part of a Masters Thesis.

(Data Science and Computational Complexity, University of Groningen)
