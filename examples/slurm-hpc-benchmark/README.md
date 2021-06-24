# slurm-hpc-benchmark

Example setup for running experiments on a [SLURM](https://slurm.schedmd.com/) cluster, using Hydra [Submitit Launcher](https://hydra.cc/docs/plugins/submitit_launcher).

## Install

Clone or copy this directory to your cluster. Then, in this directory, run:
```shell
pip install -r requirements.txt --user
```

## Usage

****. In this directory, run:
```shell
fseval --config-dir ./conf \
    --multirun \
    +experiment=my_experiment \
    +hpc=slurm \
    +dataset="glob(*)" \
    +estimator@ranker="glob(*)" \
    +estimator@validator=decision_tree,knn
```
→ runs all rankers on all datasets, and validates with Decision Tree and k-NN. Sends jobs to SLURM.

```shell

```
