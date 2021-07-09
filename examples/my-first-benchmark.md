# your-first-benchmark

Example benchmarking setup. Loads a custom ranker, by installing a pip package. Runs 20 bootstraps on 4 CPU's and then uploads the results to wandb.

## Install

In this directory, run:

```text
pip install -r requirements.txt
```

## Usage

**Locally**. In this directory, run:

```text
fseval --config-dir ./conf \
    --multirun \
    +experiment=my_experiment \
    +dataset="glob(*)" \
    +estimator@ranker=surf \
    +estimator@validator=decision_tree,knn
```

→ runs 'SURF' ranker on all datasets, and validates with Decision Tree and k-NN.

```text
[2021-06-24 10:26:26,456][HYDRA] Launching 50 jobs locally
[2021-06-24 10:26:26,456][HYDRA]     #0 : +experiment=my_experiment +dataset=boston +estimator@ranker=surf +estimator@validator=decision_tree
[2021-06-24 10:26:26,456][HYDRA]     #1 : +experiment=my_experiment +dataset=chen_additive +estimator@ranker=surf +estimator@validator=decision_tree
[2021-06-24 10:26:26,456][HYDRA]     #2 : +experiment=my_experiment +dataset=chen_orange +estimator@ranker=surf +estimator@validator=decision_tree
[2021-06-24 10:26:26,456][HYDRA]     #3 : +experiment=my_experiment +dataset=chen_xor +estimator@ranker=surf +estimator@validator=decision_tree
...
```

→ to launch on HPC systems, see other [examples](https://github.com/dunnkers/fseval/tree/master/examples).

