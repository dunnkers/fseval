# fseval

[![build status](https://github.com/dunnkers/fseval/actions/workflows/python-app.yml/badge.svg)](https://github.com/dunnkers/fseval/actions/workflows/python-app.yml) [![pypi badge](https://img.shields.io/pypi/v/fseval.svg?maxAge=3600)](https://pypi.org/project/fseval/)

A Feature Selector and Feature Ranker benchmarking library. Neatly integrates with wandb and sklearn. Uses Hydra as a config parser.

## Usage
```shell
pip install fseval
```

fseval help:
```shell
fseval --help
```

Now, create a [wandb](https://wandb.ai/) account and login to the CLI. We are now able to run benchmarks 💪🏻. The results will automatically be uploaded to the wandb dashboard.

Run ReliefF on Iris dataset:
```shell
fseval dataset=iris estimator@pipeline.ranker=relieff
```


### About
Built by [Jeroen Overschie](https://dunnkers.com/) as part of the Masters Thesis (_Data Science and Computational Complexity_ track at the University of Groningen).