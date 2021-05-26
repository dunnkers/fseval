# fseval
A Feature Selection benchmarking library. Neatly integrates with wandb and sklearn. Uses Hydra as a config parser.

## Usage
```shell
pip install git+https://github.com/dunnkers/fseval.git
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
Built by [Jeroen Overschie](https://dunnkers.com/) as part of the Masters Thesis for the Data Science and Computational Complexity track at the University of Groningen.