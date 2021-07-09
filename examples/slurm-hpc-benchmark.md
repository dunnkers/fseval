# slurm-hpc-benchmark

Example setup for running experiments on a [SLURM](https://slurm.schedmd.com/) cluster, using Hydra [Submitit Launcher](https://hydra.cc/docs/plugins/submitit_launcher).

## Install

SSH into your cluster, and clone fseval:

```text
git clone https://github.com/dunnkers/fseval.git
```

Then, install the dependencies:

```text
cd fseval/examples/slurm-hpc-benchmark/
pip install -r requirements.txt --user
```

## Usage

On your **cluster**, make sure you are in `slurm-hpc-benchmark`. Then to run _all_ rankers and _all_ datasets, run:

```text
fseval --config-dir ./conf \
    --multirun \
    +experiment=my_experiment \
    +hpc=slurm \
    +dataset="glob(*)" \
    +estimator@ranker="glob(*)" \
    +estimator@validator=decision_tree,knn
```

Your jobs are now submitted to SLURM ✨. You can CTRL+C out of the job submission screen. Sit back and watch your jobs run.

```text
[2021-06-24 12:39:24,604][HYDRA] Submitit 'slurm' sweep output dir : multirun/2021-06-24/12-39-22
[2021-06-24 12:39:24,607][HYDRA]     #0 : +experiment=my_experiment +hpc=slurm +dataset=iris +estimator@ranker=relieff +estimator@validator=decision_tree
[2021-06-24 12:39:24,621][HYDRA]     #1 : +experiment=my_experiment +hpc=slurm +dataset=iris +estimator@ranker=relieff +estimator@validator=knn
```

