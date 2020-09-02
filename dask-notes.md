# Dask & Peregrine
Notes on making Dask and Peregrine work together.

## Setup

1. To use Dask on Peregrine, we can submit local `.py` files on the cluster by loading Dask using
`module load dask`. e.g. see [example](https://jobqueue.dask.org/en/latest/examples.html#slurm-deployments). If we are in a Jupyter Lab environment we can use pip to install Dask, e.g.
`pip install dask_jobqueue` or `pip install dask`.
2. To use Jupyter Lab with the [Dask extension](https://pypi.org/project/dask-labextension/), make sure to create a new Conda environment, to
then be able to install the required packages using `jupyter labextension install dask_labextension`. See [this blogpost](https://blog.dask.org/2019/08/28/dask-on-summit) for setting up Jupyter Lab with Dask.
3. By default, the Dask Labextension is configured to create single-machine [*LocalCluster*](https://docs.dask.org/en/latest/setup/single-distributed.html#localcluster) type of clusters.
Change this by editing `~/.config/dask/labextension.yaml` (see [docs](https://docs.dask.org/en/latest/configuration.html#configuration)), to fit your SLURM configuration.

e.g.
```yaml
labextension:
   factory:
     module: 'dask_jobqueue'
     class: 'SLURMCluster'
     args: []
     kwargs: {
        'cores': 4,
        'memory': '4GB',
        'project': 'fseval',
        'interface': 'ib0'
     }
   default:
     workers: null
     adapt: null
       # minimum: 0
       # maximum: 10
   initial:
      - name: "fseval cluster"
        workers: 4
    #   - name: "Adaptive Cluster"
    #     adapt:
    #       maximum: 5
    #       minimum: 0
```

Update dashboard URL in `~/.config/dask/distributed.yaml`:


Possibly, add git functionality to Jupyter Lab: 
1. `module load git`
2. `conda install -c conda-forge jupyterlab-git`.

Possibly, add server proxying functionality to Jupyter Lab: 
1. `conda install jupyter-server-proxy -c conda-forge`.
🙌 When starting a cluster the dashboard is now visible at http://localhost:8888/proxy/8787/status