# basic-example-structured-configs
Compares two rankers on two datasets, using Hydra [Structured Configs](https://hydra.cc/docs/tutorials/structured_config/intro/).

Run all datasets and all rankers using:

```shell
python benchmark.py validator=knn dataset='glob(*)' ranker='glob(*)' --multirun
```

<!-- TODO: add interactive CLI log / screenshot. -->