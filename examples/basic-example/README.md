# basic-example
Compares two rankers on two datasets, using [Hydra](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/) and fseval.

Run all datasets and all rankers using:

```shell
python benchmark.py validator=knn dataset='glob(*)' ranker='glob(*)' --multirun
```

<!-- TODO: add interactive CLI log / screenshot. -->