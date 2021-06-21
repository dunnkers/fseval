# fseval

[![build status](https://github.com/dunnkers/fseval/actions/workflows/python-app.yml/badge.svg)](https://github.com/dunnkers/fseval/actions/workflows/python-app.yml) [![pypi badge](https://img.shields.io/pypi/v/fseval.svg?maxAge=3600)](https://pypi.org/project/fseval/)

A Feature Selector and Feature Ranker benchmarking library. Neatly integrates with [wandb](https://wandb.ai) and [sklearn](https://scikit-learn.org/). Uses [Hydra](https://hydra.cc/) as a config parser.

## Usage
```shell
pip install fseval
```

fseval help:
```shell
fseval --help
```

Now, create a [wandb](https://wandb.ai/) account and login to the CLI. We are now able to run benchmarks 💪🏻. The results will automatically be uploaded to the wandb dashboard.

Run ANOVA F-Value on Iris dataset:
```shell
fseval +dataset=iris +estimator@ranker=anova_f_value +estimator@validator=decision_tree
```

## Supported Feature Rankers
A [collection](https://github.com/dunnkers/fseval/tree/master/fseval/conf/estimator) of feature rankers are already built-in, which can be used without further configuring. Others need their dependencies installed. List of rankers:

| Ranker | Dependency | Command line argument
--- | --- | ---
[ANOVA F-Value](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.f_classif) | \<no dep\> | `estimator@ranker=anova_f_value`
[Boruta](https://github.com/scikit-learn-contrib/boruta_py) | `pip install Boruta` | `estimator@ranker=boruta`
[Chi2](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html) | \<no dep\> | `estimator@ranker=chi2`
[Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) | \<no dep\> | `estimator@ranker=decision_tree`
[FeatBoost](https://github.com/amjams/FeatBoost) | `pip install git+https://github.com/dunnkers/FeatBoost.git@support-cloning` (ℹ️) | `estimator@ranker=featboost`
[MultiSURF](https://github.com/EpistasisLab/scikit-rebate) | `pip install skrebate` | `estimator@ranker=multisurf`
[Mutual Info](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html) | \<no dep\> | `estimator@ranker=mutual_info`
[ReliefF](https://github.com/EpistasisLab/scikit-rebate) | `pip install skrebate` | `estimator@ranker=relieff`
[Stability Selection](https://github.com/scikit-learn-contrib/stability-selection) | `pip install git+https://github.com/dunnkers/stability-selection.git@master matplotlib` (ℹ️) | `estimator@ranker=stability_selection`
[TabNet](https://github.com/dreamquark-ai/tabnet) | `pip install pytorch-tabnet` | `estimator@ranker=tabnet`
[XGBoost](https://xgboost.readthedocs.io/) | `pip install xgboost` | `estimator@ranker=xgb`
[Infinite Selection](https://github.com/giorgioroffo/Infinite-Feature-Selection) | `pip install git+https://github.com/dunnkers/infinite-selection.git@master` (ℹ️) | `estimator@ranker=infinite_selection`


ℹ️ This library was customized to make it compatible with the fseval pipeline.

If you would like to install simply all dependencies, download the fseval [requirements.txt](https://github.com/dunnkers/fseval/blob/master/requirements.txt) file and run `pip install -r requirements.txt`.

## Wandb support
Wandb can be enabled by using `+backend=wandb`. It's used to store metrics, but also files. Set any parameter to be passed to `wandb.init` like so:

```shell
fseval callbacks.wandb.project=<your-project-name> callbacks.wandb.group=<run-group>
```

Runs can be restored as follows:

```shell
fseval callbacks.wandb.id=<wandb_run_id> callbacks.wandb.log_metrics=false
```
→ make sure the rest of the config is the same as the previous run. You can now overwrite tables.

To disable wandb, use:
```shell
fseval "~callbacks.wandb"
```
### About
Built by [Jeroen Overschie](https://dunnkers.com/) as part of the Masters Thesis (_Data Science and Computational Complexity_ track at the University of Groningen).