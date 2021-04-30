# mleval
A Machine Learning benchmarking library. Neatly integrates with wandb and sklearn. Uses Hydra as a config parser.



## Architecture
Uses [RQ](https://python-rq.org/) (Redis Queue) as a launcher for Hydra. For this reason we require a Redis server. Follow Hydra RQ [instructions](https://hydra.cc/docs/next/plugins/rq_launcher/): set all necessary environment variables. Also:

- Take note of whether your Redis server uses SSL or not `redis` versus `rediss`. In the case of SSL, we require extra config:

### Running Hydra & RQ:
To launch an RQ multi-run:
```shell
WANDB_MODE=dryrun python run.py hydra/launcher=rq bootstrap.random_state=0,1 --multirun
```

At any time, observe the current Hydra config using:
```shell
python run.py  --cfg hydra
```

Or the job specific config:
```shell
python run.py  --cfg job
```

Launch a dashboard using:
```shell
rq-dashboard -u $REDIS_URL
```

When getting the following error:
> We cannot safely call it or ignore it in the fork() child process. Crashing instead.
Use ([src](https://stackoverflow.com/questions/50168647/multiprocessing-causes-python-to-crash-and-gives-an-error-may-have-been-in-progr)):

```shell
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
```