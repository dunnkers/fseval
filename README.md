# fseval

[![build status](https://github.com/dunnkers/fseval/actions/workflows/python-app.yml/badge.svg)](https://github.com/dunnkers/fseval/actions/workflows/python-app.yml) [![pypi badge](https://img.shields.io/pypi/v/fseval.svg?maxAge=3600)](https://pypi.org/project/fseval/)

A Feature Selector and Feature Ranker benchmarking library. Neatly integrates with [Weights and Biases](https://wandb.ai) and [Sci-kit Learn](https://scikit-learn.org/). Uses [Hydra](https://hydra.cc/) as a config parser.

## Install

```shell
pip install fseval
```

## Usage
fseval is run via a CLI. As an example, this runs a very simple benchmark:
```shell
fseval +dataset=synclf_easy +estimator@ranker=chi2 +estimator@validator=decision_tree
```

Which runs Chi2 feature ranking on the 'Iris' dataset, and validates feature subsets using k-NN. The results can be uploaded to a backend. We can use **wandb** for this.


### Integration with wandb
Integration with [wandb](https://wandb.ai) is built-in. Create an account and login to the [CLI](https://github.com/wandb/client#-simple-integration-with-any-framework) with `wandb login`. Then, we can upload results to wandb using `+callbacks="[wandb]"`, like so:

```shell
fseval +dataset=synclf_easy +estimator@ranker=chi2 +estimator@validator=decision_tree callbacks="[wandb]" +callbacks.wandb.project=fseval-readme
```

This runs an experiment and uploads the results to wandb:
<p align="center">
  <img width="800" src="./docs/run-cli-example.svg">
</p>


We can now explore the results on the online dashboard:

[![run-wandb-example](./docs/run-wandb-example.png)](https://wandb.ai/dunnkers/fseval-readme/runs/11b4t26e)

✨


To see all the configurable options, run:
```shell
fseval --help
```

### Running bootstraps
_Bootstraps_ can be run, to approximate the stability of an algorithm. Bootstrapping works by creating multiple dataset permutations and running the algorithm on each of them. A simple way to create dataset permutations is to resample **with replacement**.

In fseval, bootstrapping can be configured like so:

```shell
fseval [...] **resample=bootstrap n_bootstraps=8**
```

To run the entire experiment 8 times, each for a resampled dataset. Ideally, when multiple processors are used, the number of bootstraps is set to an amount that is divisible by the amount of CPU's. For example:

```shell
fseval [...] resample=bootstrap n_bootstraps=8 **n_jobs=4**
```

would cause all 8 CPU's to be utilized efficiently.

When using bootstraps, all results in the dashboard will be aggregated over all bootstraps. ✨

## Built-in Feature Rankers
A [collection](https://github.com/dunnkers/fseval/tree/master/fseval/conf/estimator) of feature rankers are already built-in, which can be used without further configuring. Others need their dependencies installed. List of rankers:

| Ranker | Dependency | Command line argument
--- | --- | ---
[ANOVA F-Value](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.f_classif) | \<no dep\> | `+estimator@ranker=anova_f_value`
[Boruta](https://github.com/scikit-learn-contrib/boruta_py) | `pip install Boruta` | `+estimator@ranker=boruta`
[Chi2](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html) | \<no dep\> | `+estimator@ranker=chi2`
[Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) | \<no dep\> | `+estimator@ranker=decision_tree`
[FeatBoost](https://github.com/amjams/FeatBoost) | `pip install git+https://github.com/amjams/FeatBoost.git` | `+estimator@ranker=featboost`
[MultiSURF](https://github.com/EpistasisLab/scikit-rebate) | `pip install skrebate` | `+estimator@ranker=multisurf`
[Mutual Info](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html) | \<no dep\> | `+estimator@ranker=mutual_info`
[ReliefF](https://github.com/EpistasisLab/scikit-rebate) | `pip install skrebate` | `+estimator@ranker=relieff`
[Stability Selection](https://github.com/scikit-learn-contrib/stability-selection) | `pip install git+https://github.com/dunnkers/stability-selection.git matplotlib` (ℹ️) | `+estimator@ranker=stability_selection`
[TabNet](https://github.com/dreamquark-ai/tabnet) | `pip install pytorch-tabnet` | `+estimator@ranker=tabnet`
[XGBoost](https://xgboost.readthedocs.io/) | `pip install xgboost` | `+estimator@ranker=xgb`
[Infinite Selection](https://github.com/giorgioroffo/Infinite-Feature-Selection) | `pip install git+https://github.com/dunnkers/infinite-selection.git` (ℹ️) | `+estimator@ranker=infinite_selection`


ℹ️ This library was customized to make it compatible with the fseval pipeline.

If you would like to install simply all dependencies, download the fseval [requirements.txt](https://github.com/dunnkers/fseval/blob/master/requirements.txt) file and run `pip install -r requirements.txt`.

### About
Built by [Jeroen Overschie](https://dunnkers.com/) as part of the Masters Thesis (_Data Science and Computational Complexity_ track at the University of Groningen).