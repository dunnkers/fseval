import tempfile
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest
import wandb
from fseval.storage.wandb import WandbStorage
from fseval.utils.uuid_utils import generate_shortuuid
from pytest import Cache, FixtureRequest
from wandb.apis.public import Api, File, Run

ENTITY: str = "fseval"
PROJECT: str = "test_wandb.py"


@pytest.mark.dependency()
def test_save(request: FixtureRequest):
    run_id: str = generate_shortuuid()
    wandb.init(entity=ENTITY, project=PROJECT, id=run_id)

    # write text to some file
    assert wandb.run is not None, "wandb not initialized"
    save_dir: str = wandb.run.dir
    filename: str = "some_file.txt"
    filepath: Path = Path(save_dir) / filename

    # save some object
    df: pd.DataFrame = pd.DataFrame([{"acc": 0.8}])
    wandb_storage: WandbStorage = WandbStorage()
    wandb_storage.save(filename=filename, writer=df.to_csv)

    # finish run
    wandb.finish()

    # 🧪 store run in pytest cache
    cache = request.config.cache
    assert cache is not None
    cache.set("run_id", run_id)
    cache.set("filename", filename)


@pytest.mark.dependency(depends=["test_save"])
def test_load(request: FixtureRequest):
    # retrieve previous run id from cache
    cache = request.config.cache
    assert cache is not None
    run_id: str = cache.get("run_id", None)
    filename: str = cache.get("filename", None)

    # start wandb and try to restore file from previous run
    wandb.init(entity=ENTITY, project=PROJECT)
    wandb_storage: WandbStorage = WandbStorage(run_id=run_id)
    reader: Callable = lambda file_handle: pd.read_csv(file_handle, index_col=0)
    df: pd.DataFrame = wandb_storage.restore(filename, reader)
    assert df is not None
    assert len(df.columns) == 1
    assert "acc" in df.columns
    assert df["acc"][0] == 0.8

    # Finish run
    assert wandb.run is not None
    run_id_current = wandb.run.id
    wandb.finish()

    # 🧹 Clean up
    api: Api = wandb.Api()
    run: Run = api.run(path=f"{ENTITY}/{PROJECT}/{run_id}")
    run.delete(delete_artifacts=True)
    run = api.run(path=f"{ENTITY}/{PROJECT}/{run_id_current}")
    run.delete(delete_artifacts=True)
