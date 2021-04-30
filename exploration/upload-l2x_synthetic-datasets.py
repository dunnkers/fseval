
#%%
import wandb
import numpy as np
from slugify import slugify
from l2x_synthetic.make_data import generate_data

#%%
def store_dataset_in_artifact(name, X, Y):
    artifact = wandb.Artifact(slugify(name), type='dataset')
    
    n, p = np.shape(X)
    columns = [f'X{p}' for p in range(1, p + 1)]
    table = wandb.Table(columns=columns, data=X)
    artifact.add(table, 'X')

    n, p = np.shape(Y)
    columns = [f'Y{p}' for p in range(1, p + 1)]
    table = wandb.Table(columns=columns, data=Y)
    artifact.add(table, 'Y')

    run.log_artifact(artifact)

#%%
run = wandb.init(job_type="dataset-creation")

X, Y = generate_data(n=10000, datatype='orange_skin', seed=0)
store_dataset_in_artifact('orange_skin', X, Y)

X, Y = generate_data(n=10000, datatype='nonlinear_additive', seed=0)
store_dataset_in_artifact('nonlinear_additive', X, Y)

X, Y = generate_data(n=10000, datatype='XOR', seed=0)
store_dataset_in_artifact('XOR', X, Y)

X, Y = generate_data(n=10000, datatype='switch', seed=0)
store_dataset_in_artifact('switch', X, Y)

wandb.finish()