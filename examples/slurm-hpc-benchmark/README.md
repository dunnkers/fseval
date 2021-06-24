# slurm-hpc-benchmark

Example setup for running experiments on a [SLURM](https://slurm.schedmd.com/) cluster, using Hydra [Submitit Launcher](https://hydra.cc/docs/plugins/submitit_launcher).

## Install

SSH into your cluster, and clone fseval:
```shell
git clone https://github.com/dunnkers/fseval.git
```

Then, install the dependencies:
```shell
cd fseval/examples/slurm-hpc-benchmark/
pip install -r requirements.txt --user
```

## Usage

On your **cluster**, make sure you are in `slurm-hpc-benchmark`. Then to run _all_ rankers and _all_ datasets, run:
```shell
fseval --config-dir ./conf \
    --multirun \
    +experiment=my_experiment \
    +hpc=slurm \
    +dataset="glob(*)" \
    +estimator@ranker="glob(*)" \
    +estimator@validator=decision_tree,knn
```

Your jobs are now submitted to SLURM ✨. You can CTRL+C out of the job submission screen. Sit back and watch your jobs run.
