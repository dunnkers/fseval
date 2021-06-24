# slurm-hpc-benchmark

Example setup for running experiments on a [SLURM](https://slurm.schedmd.com/) cluster, using Hydra [Submitit Launcher](https://hydra.cc/docs/plugins/submitit_launcher).

## Install

Clone fseval on your cluster, and cd into this example:
```shell
git clone https://github.com/dunnkers/fseval.git
cd fseval/examples/slurm-hpc-benchmark/
```

Then, run:
```shell
pip install -r requirements.txt --user
```

## Usage

Cd into the `slurm-hpc-benchmark` directory. On your **cluster**, run:
```shell
fseval --config-dir ./conf \
    --multirun \
    +experiment=my_experiment \
    +hpc=slurm \
    +dataset="iris" \
    +estimator@ranker="relieff" \
    +estimator@validator=decision_tree,knn
```
→ runs all rankers on all datasets, and validates with Decision Tree and k-NN. Sends jobs to SLURM.

```shell

```
