# Welcome!

![](.gitbook/assets/header.png)

## fseval

[![build status](https://github.com/dunnkers/fseval/actions/workflows/python-app.yml/badge.svg)](https://github.com/dunnkers/fseval/actions/workflows/python-app.yml) [![pypi badge](https://img.shields.io/pypi/v/fseval.svg?maxAge=3600)](https://pypi.org/project/fseval/)

A Feature Ranker benchmarking library. Useful for Feature Selection and Interpretable AI methods. Allows plotting feature importance scores on an online dashboard. Neatly integrates [Hydra](https://hydra.cc/) with [wandb](https://wandb.ai).

Any [sklearn](https://scikit-learn.org/) style estimator can be used as a Feature Ranker. Estimator must estimate at least one of:

1. **Feature importance**, using `feature_importances_`.
2. **Feature subset**, using `feature_support_`.
3. **Feature ranking**, using `feature_ranking_`.

Main functionality:

* 📊 **Online dashboard**. Experiments can be uploaded to [wandb](https://wandb.ai) for seamless experiment tracking and visualization. Feature importance and subset validation plots are built-in. 
* 🔄 **Scikit-Learn integration**. Integrates nicely with [sklearn](https://scikit-learn.org/). Any estimator that implements `fit` is supported.
* 🗄 **Dataset adapters**. Datasets can be loaded dynamically using an _adapter_. [OpenML](https://www.openml.org/search?type=data) support is built-in.
* 🎛 **Synthetic dataset generation**. Synthetic datasets can be generated and configured right in the library itself.
* 📌 **Relevant features ground-truth**. Datasets can have ground-truth relevant features defined, so the estimated versus the ground-truth feature importance is automatically plotted in the dashboard.
* ⚜️ **Subset validation**. Allows you to validate the quality of a feature ranking, by running a _validation_ estimator on some of the `k` best feature subsets.
* ⚖️ **Bootstrapping**. Allows you to approximate the _stability_ of an algorithm by running multiple experiments on bootstrap resampled datasets.
* ⚙️ **Reproducible configs**. Uses [Hydra](https://hydra.cc/) as a config parser, to allow configuring every part of the experiment. The config can be uploaded to wandb, so the experiment can be replayed later.

### Install

```text
pip install fseval
```

### Usage

fseval is run via a CLI. Example:

```text
fseval \
  +dataset=synclf_easy \
  +estimator@ranker=chi2 \
  +estimator@validator=decision_tree
```

Which runs [Chi2](https://github.com/dunnkers/fseval/blob/master/fseval/conf/estimator/chi2.yaml) feature ranking on the [Synclf easy](https://github.com/dunnkers/fseval/blob/master/fseval/conf/dataset/synclf_easy.yaml) dataset, and validates feature subsets using a [Decision Tree](https://github.com/dunnkers/fseval/blob/master/fseval/conf/estimator/decision_tree.yaml).

![](.gitbook/assets/run-cli-example.svg)

To see all the configurable options, run:

```text
fseval --help
```

#### Weights and Biases integration

Integration with [wandb](https://wandb.ai) is built-in. Create an account and login to the [CLI](https://github.com/wandb/client#-simple-integration-with-any-framework) with `wandb login`. Then, enable wandb using `callbacks="[wandb]"`:

```text
fseval \
  callbacks="[wandb]" \
  +callbacks.wandb.project=fseval-readme \
  +dataset=synclf_easy \
  +estimator@ranker=chi2 \
  +estimator@validator=decision_tree
```

We can now explore the results on the online dashboard:

 [![](.gitbook/assets/run-wandb-example.png)](https://wandb.ai/dunnkers/fseval-readme/runs/11b4t26e)

✨

#### Running bootstraps

_Bootstraps_ can be run, to approximate the stability of an algorithm. Bootstrapping works by creating multiple dataset permutations and running the algorithm on each of them. A simple way to create dataset permutations is to **resample with replacement**.

In fseval, bootstrapping can be configured with `resample=bootstrap`:

```text
fseval \
  resample=bootstrap \
  n_bootstraps=8 \
  +dataset=synclf_easy \
  +estimator@ranker=chi2 \
  +estimator@validator=decision_tree
```

To run the entire experiment 8 times, each for a resampled dataset.

In the dashboard, plots are already set up to support bootstrapping:

![](.gitbook/assets/run-bootstraps-example.svg)

Shows the validation results for **25** bootstraps. ✨

#### Launching multiple experiments at once

To launch multiple experiments, use [`--multirun`](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run).

```text
fseval \
  --multirun \
  +dataset=synclf_easy \
  +estimator@ranker=[boruta,featboost,chi2] \
  +estimator@validator=decision_tree
```

Which launches 3 jobs.

See the multirun overriding [syntax](https://hydra.cc/docs/advanced/override_grammar/extended). For example, you can select multiple groups using `[]`, a range using `range(start, stop, step)` and all options using `glob(*)`.

#### Multiprocessing

The experiment can run in parallel. The list of bootstraps is distributed over the CPU's. To use all available processors set `n_jobs=-1`:

```text
fseval [...] n_jobs=-1
```

Alternatively, set `n_jobs` to the specific amount of processors to use. e.g. `n_jobs=4` if you have a quad-core.

When using bootstraps, it can be efficient to use an amount that is divisible by the amount of CPU's:

```text
fseval [...] resample=bootstrap n_bootstraps=8 n_jobs=4
```

would cause all 8 CPU's to be utilized efficiently.

#### Distributed processing

Since fseval uses Hydra, all its plugins can also be used. Some plugins for distributed processing are:

* [RQ launcher](https://hydra.cc/docs/plugins/rq_launcher/). Uses Redis Queue \([RQ](https://python-rq.org/)\) to launch jobs.
* [Submitit Launcher](https://hydra.cc/docs/plugins/submitit_launcher). Submits jobs directly to a [SLURM](https://slurm.schedmd.com/) cluster. See the [example setup](https://github.com/dunnkers/fseval/tree/master/examples/slurm-hpc-benchmark).

Example:

```text
fseval --multirun [...] hydra/launcher=rq
```

To submit jobs to RQ.

#### Configuring a Feature Ranker

The entirety of the config can be overriden like pleased. Like such, also feature rankers can be configured. For example:

```text
fseval [...] +validator.classifier.estimator.criterion=entropy
```

Changes the Decision Tree criterion to entropy. One could perform a **hyper-parameter sweep** over some parameters like so:

```text
fseval --multirun [...] +validator.classifier.estimator.criterion=entropy,gini
```

or, in case of a ranker:

```text
fseval --multirun [...] +ranker.classifier.estimator.learning_rate="range(0.1, 2.1, 0.1)"
```

Which launches 20 jobs with different learning rates \(this hyper-parameter applies to `+estimator@ranker=featboost`\). See [multirun](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run) docs for syntax.

### Config directory

The amount of command-line arguments quickly adds up. Any configuration can also be loaded from a dir. It is configured with [`--config-dir`](https://hydra.cc/docs/advanced/hydra-command-line-flags):

```text
fseval --config-dir ./conf
```

With the `./conf` directory containing:

```text
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

```text
.
└── conf
    ├── estimator
    │   └── my_custom_ranker.yaml
    └── dataset
        └── my_custom_dataset.yaml
```

We can now use the newly installed estimator and dataset:

```text
fseval --config-dir ./conf +estimator@ranker=my_custom_ranker +dataset=my_custom_dataset
```

🙌🏻

Where `my_custom_ranker.yaml` would be any [estimator](https://github.com/dunnkers/fseval/tree/master/fseval/conf/estimator) definition, and `my_custom_dataset.yaml` any [dataset](https://github.com/dunnkers/fseval/tree/master/fseval/conf/dataset) dataset definition.

### Built-ins

Several rankers, datasets and validators are already built-in.

Built-in Feature Rankers \| Ranker \| Source \| Command \| Classif- ication \| Regr- ession \| Multi output \| Feature importance \| Feature support \| Feature ranking \| \|-\|-\|-\|-\|-\|-\|-\|-\|-\| \| ANOVA F-value \| \[sklearn\]\(https://scikit-learn.org/stable/modules/generated/sklearn.feature\_selection.f\_classif.html\)\) \|cli\`+estimator@ranker=anova\_f\_value\`\| ✓ \| ✓ \| \| ✓ \| \| \| \| Boruta \| \[github\]\(https://github.com/scikit-learn-contrib/boruta\_py\)\)install\`pip install Boruta\` \|cli\`+estimator@ranker=boruta\`\| ✓ \| \| \| \| ✓ \| ✓ \| \| Chi-Squared \| \[sklearn\]\(https://scikit-learn.org/stable/modules/generated/sklearn.feature\_selection.chi2.html\)\) \|cli\`+estimator@ranker=chi2\`\| ✓ \| \| \| ✓ \| \| \| \| Decision Tree \| \[sklearn\]\(https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html\)\) \|cli\`+estimator@ranker=decision\_tree\`\| ✓ \| ✓ \| ✓ \| ✓ \| \| \| \| FeatBoost \| \[github\]\(https://github.com/amjams/FeatBoost\)\)install\`pip install git+https://github.com/amjams/FeatBoost.git\` \|cli\`+estimator@ranker=featboost\`\| ✓ \| \| \| ✓ \| ✓ \| \| \| Infinite Selection \| \[github\]\(https://github.com/giorgioroffo/Infinite-Feature-Selection\)\)install\`pip install git+https://github.com/dunnkers/infinite-selection.git\` ℹ️ \|cli\`+estimator@ranker=infinite\_selection\`\| ✓ \| \| \| ✓ \| \| ✓ \| \| MultiSURF \| \[github\]\(https://github.com/EpistasisLab/scikit-rebate\)\)install\`pip install skrebate\` \|cli\`+estimator@ranker=multisurf\`\| ✓ \| ✓ \| \| ✓ \| \| \| \| Mutual Info \| \[github\]\(https://scikit-learn.org/stable/modules/generated/sklearn.feature\_selection.mutual\_info\_classif.html\)\) \|cli\`+estimator@ranker=mutual\_info\`\| ✓ \| ✓ \| \| ✓ \| \| \| \| ReliefF \| \[github\]\(https://github.com/EpistasisLab/scikit-rebate\)\)install\`pip install skrebate\` \|cli\`+estimator@ranker=relieff\`\| ✓ \| ✓ \| \| ✓ \| \| \| \| Stability Selection \| \[github\]\(https://github.com/scikit-learn-contrib/stability-selection\)\)install\`pip install git+https://github.com/dunnkers/stability-selection.git matplotlib\` ℹ️ \|cli\`+estimator@ranker=stability\_selection\`\| ✓ \| \| \| ✓ \| ✓ \| \| \| TabNet \| \[github\]\(https://github.com/dreamquark-ai/tabnet\)\)install\`pip install pytorch-tabnet\` \|cli\`+estimator@ranker=tabnet\`\| ✓ \| ✓ \| ✓ \| ✓ \| \| \| \| XGBoost \| \[github\]\(https://xgboost.readthedocs.io/\)\)install\`pip install xgboost\` \|cli\`+estimator@ranker=xgb\`\| ✓ \| ✓ \| \| ✓ \| \| \| ℹ️ This library was customized to make it compatible with the fseval pipeline. If you would like to install simply all dependencies, download the fseval \[requirements.txt\]\(https://github.com/dunnkers/fseval/blob/master/requirements.txt\) file and run \`pip install -r requirements.txt\`.

Built-in Datasets \| Dataset \| Source \| Command \| \`n\` \| \`p\` \| Task \| Multi- output \| Domain \| \|-------------------------------------------\|-------\|-------------\|-------------\|----------------\|------------------------\|--------------------------------------\|----------------------------------------\| \| Boston house prices \| \[OpenML\]\(https://www.openml.org/d/531\)install\`pip install openml\` \|cli\`+dataset=boston\` \| 506 \| 11 \| Regression \| No \| Finance \| \| Additive \(Chen et al. \[L2X\]\(https://github.com/Jianbo-Lab/L2X\)\) \| \[Synthetic\]\(https://github.com/dunnkers/l2x\_synthetic\)install\`pip install l2x-synthetic\` \|cli\`+dataset=chen\_additive\` \| 10000 \| 10 \| Regression \| Yes \| Synthetic \| \| Orange \(Chen et al. \[L2X\]\(https://github.com/Jianbo-Lab/L2X\)\) \| \[Synthetic\]\(https://github.com/dunnkers/l2x\_synthetic\)install\`pip install l2x-synthetic\` \|cli\`+dataset=chen\_orange\` \| 10000 \| 10 \| Regression \| Yes \| Synthetic \| \| XOR \(Chen et al. \[L2X\]\(https://github.com/Jianbo-Lab/L2X\)\) \| \[Synthetic\]\(https://github.com/dunnkers/l2x\_synthetic\)install\`pip install l2x-synthetic\` \|cli\`+dataset=chen\_xor\` \| 10000 \| 10 \| Regression \| Yes \| Synthetic \| \| Climate Model Simulation \| \[OpenML\]\(https://www.openml.org/d/40994\) \(\[CC18\]\(https://docs.openml.org/benchmark/\#openml-cc18\)\)install\`pip install openml\` \|cli\`+dataset=climate\_model\_simulation\` \| 540 \| 18 \| Classification \| No \| Nature \| \| Cylinder bands \| \[OpenML\]\(https://www.openml.org/d/1497\) \(\[CC18\]\(https://docs.openml.org/benchmark/\#openml-cc18\)\)install\`pip install openml\` \|cli\`+dataset=cylinder\_bands\` \| 5456 \| 24 \| Classification \| No \| Mechanics \| \| Iris Flowers \| \[OpenML\]\(https://www.openml.org/d/61\)install\`pip install openml\` \|cli\`+dataset=iris\` \| 150 \| 4 \| Classification \| No \| Nature \| \| Madelon \| \[OpenML\]\(https://www.openml.org/d/1485\)install\`pip install openml\` \|cli\`+dataset=madelon\` \| 2600 \| 500 \| Classification \| No \| Synthetic \| \| Multifeat Pixel \| \[OpenML\]\(https://www.openml.org/d/40979\) \(\[CC18\]\(https://docs.openml.org/benchmark/\#openml-cc18\)\)install\`pip install openml\` \|cli\`+dataset=mfeat\_pixel\` \| 2000 \| 240 \| Classification \| No \| OCR \| \| Nomao \| \[OpenML\]\(https://www.openml.org/d/1486\) \(\[CC18\]\(https://docs.openml.org/benchmark/\#openml-cc18\)\)install\`pip install openml\` \|cli\`+dataset=nomao\` \| 34465 \| 89 \| Classification \| No \| Geodata \| \| Ozone Levels \| \[OpenML\]\(https://www.openml.org/d/1487\) \(\[CC18\]\(https://docs.openml.org/benchmark/\#openml-cc18\)\)install\`pip install openml\` \|cli\`+dataset=ozone\_levels\` \| 2534 \| 72 \| Classification \| No \| Nature \| \| Phoneme \| \[OpenML\]\(https://www.openml.org/d/1489\) \(\[CC18\]\(https://docs.openml.org/benchmark/\#openml-cc18\)\)install\`pip install openml\` \|cli\`+dataset=phoneme\` \| 5404 \| 5 \| Classification \| No \| Biomedical \| \| Synclf easy \| \[Synthetic\]\(https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make\_classification.html\) \|cli\`+dataset=synclf\_easy\` \| 10000 \| 20 \| Classification \| No \| Synthetic \| \| Synclf medium \| \[Synthetic\]\(https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make\_classification.html\) \|cli\`+dataset=synclf\_medium\` \| 10000 \| 30 \| Classification \| No \| Synthetic \| \| Synclf hard \| \[Synthetic\]\(https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make\_classification.html\) \|cli\`+dataset=synclf\_hard\` \| 10000 \| 50 \| Classification \| No \| Synthetic \| \| Synclf very hard \| \[Synthetic\]\(https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make\_classification.html\) \|cli\`+dataset=synclf\_very\_hard\` \| 10000 \| 50 \| Classification \| No \| Synthetic \| \| Synreg easy \| \[Synthetic\]\(https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make\_regression.html\) \|cli\`+dataset=synreg\_easy\` \| 10000 \| 10 \| Regression \| No \| Synthetic \| \| Synreg medium \| \[Synthetic\]\(https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make\_regression.html\) \|cli\`+dataset=synreg\_medium\` \| 10000 \| 10 \| Regression \| No \| Synthetic \| \| Synreg hard \| \[Synthetic\]\(https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make\_regression.html\) \|cli\`+dataset=synreg\_hard\` \| 10000 \| 20 \| Regression \| No \| Synthetic \| \| Synreg hard \| \[Synthetic\]\(https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make\_regression.html\) \|cli\`+dataset=synreg\_very\_hard\` \| 10000 \| 20 \| Regression \| No \| Synthetic \| \| Texture \| \[OpenML\]\(https://www.openml.org/d/40499\) \(\[CC18\]\(https://docs.openml.org/benchmark/\#openml-cc18\)\)install\`pip install openml\` \|cli\`+dataset=texture\` \| 5500 \| 40 \| Classification \| No \| Pattern Recognition \| \| Wall Robot Navigation \| \[OpenML\]\(https://www.openml.org/d/1497\) \(\[CC18\]\(https://docs.openml.org/benchmark/\#openml-cc18\)\)install\`pip install openml\` \|cli\`+dataset=wall\_robot\_navigation\` \| 5456 \| 24 \| Classification \| No \| Mechanics \| - \`n\`: number of dataset \*\*samples\*\*. - \`p\`: number of dataset \*\*dimensions\*\*.

Built-in Validators \| Validator \| Source \| Command \| Classification \| Regression \| Multioutput \| \|-\|-\|-\|-\|-\|-\| \| Decision Tree \| \[sklearn\]\(https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html\) \|cli\`+estimator@validator=decision\_tree\`\| ✓ \| ✓ \| ✓ \| \| k-NN \| \[sklearn\]\(https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html\) \|cli\`+estimator@validator=knn\`\| ✓ \| \| \| \| TabNet \| \[github\]\(https://github.com/dreamquark-ai/tabnet\)install\`pip install pytorch-tabnet\` \|cli\`+estimator@validator=tabnet\`\| ✓ \| ✓ \| ✓ \| \| XGBoost \| \[github\]\(https://xgboost.readthedocs.io/\)install\`pip install xgboost\` \|cli\`+estimator@validator=xgb\`\| ✓ \| ✓ \| \|

ℹ️ Note you _cannot_ mix built-ins and custom rankers/datasets/validators in a **multirun**. This is due to the behavior of the [Hydra](https://github.com/facebookresearch/hydra) library.

### About

Built by at the University of Groningen.

2021 — [Jeroen Overschie](https://dunnkers.com/)

