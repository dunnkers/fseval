import pandas as pd
import pytest
import wandb
from fseval.callbacks.wandb import WandbCallback
from fseval.utils.dict_utils import dict_merge
from fseval.utils.uuid_utils import generate_shortuuid
from omegaconf import DictConfig, OmegaConf
from pytest import FixtureRequest
from wandb.apis.public import Api, Run
from wandb.errors import CommError

ENTITY: str = "fseval"
PROJECT: str = "test_wandb.py"
RUN_CONFIG_1 = {"some_setting": "which should be saved"}
RUN_CONFIG_2 = {"some_other_setting": 41}


@pytest.fixture
def api() -> Api:
    api: Api = wandb.Api()
    return api


@pytest.fixture
def wandb_callback() -> WandbCallback:
    wandb_callback = WandbCallback(entity=ENTITY, project=PROJECT)
    return wandb_callback


def test_init():
    # with logging metrics
    wandb_callback = WandbCallback()
    assert wandb_callback.log_metrics == True
    assert wandb_callback.callback_config == {}

    # disable metrics
    wandb_callback = WandbCallback(log_metrics=False)
    assert wandb_callback.log_metrics == False


def test_on_begin_parameters():
    wandb_callback = WandbCallback(some_unknown_metric=False)

    # verify a TypeError is raised when unknown parameters are passed
    with pytest.raises(TypeError):
        wandb_callback.on_begin(OmegaConf.create({}))


@pytest.mark.dependency()
def test_on_begin(wandb_callback: WandbCallback, api: Api, request: FixtureRequest):
    run_id: str = generate_shortuuid()
    wandb_callback.callback_config["id"] = run_id

    # run `on_begin`
    config: DictConfig = OmegaConf.create(RUN_CONFIG_1)
    wandb_callback.on_begin(config)

    # trying to fetch an unknown run should raise this error
    with pytest.raises(CommError):
        api.run(path="some_entity/some_project/some_non_existent_run")

    # assert whether run indeed exists with `run_id`
    # raises a `CommError when not existent`
    run: Run = api.run(path=f"{ENTITY}/{PROJECT}/{run_id}")
    assert run.state == "running"
    assert run.id == run_id
    assert run.config == RUN_CONFIG_1

    # 🧪 store run in pytest cache
    cache = request.config.cache
    assert cache is not None
    cache.set("run_id", run_id)


@pytest.mark.dependency(depends=["test_on_begin"])
def test_on_config_update(wandb_callback: WandbCallback):
    wandb_callback.on_config_update(RUN_CONFIG_2)

    # → assertions in `test_on_end`


@pytest.mark.dependency(depends=["test_on_begin"])
def test_on_metrics(wandb_callback: WandbCallback):
    # should not be able to log ints. only dicts.
    with pytest.raises(ValueError):
        wandb_callback.on_metrics(123123)  # type: ignore

    # upload metric
    wandb_callback.on_metrics({"acc": 0.99})

    # this metric should not upload
    wandb_callback.log_metrics = False
    wandb_callback.on_metrics({"should_not_upload": 999})

    # → assertions in `test_on_end`


@pytest.mark.dependency(depends=["test_on_begin"])
def test_on_table(wandb_callback: WandbCallback):
    # upload simple weather table
    df = pd.DataFrame([{"temperature": 23.0}])
    wandb_callback.on_table(df=df, name="weather")

    # → assertions in `test_on_end`


@pytest.mark.dependency(depends=["test_on_begin"])
def test_on_summary(wandb_callback: WandbCallback):
    wandb_callback.on_summary({"avg_acc": 0.8})

    # → assertions in `test_on_end`


@pytest.mark.dependency(
    depends=[
        "test_on_begin",
        "test_on_config_update",
        "test_on_metrics",
        "test_on_table",
        "test_on_summary",
    ]
)
def test_on_end(wandb_callback: WandbCallback, api: Api, request: FixtureRequest):
    # 🧪 retrieve cache
    cache = request.config.cache
    assert cache is not None
    run_id: str = cache.get("run_id", None)

    # `on_end`
    wandb_callback.on_end()

    # ensure finished
    api._runs = {}  # reset wandb runs internal cache
    run: Run = api.run(path=f"{ENTITY}/{PROJECT}/{run_id}")
    run_state: str = run.state
    assert run_state == "finished"

    ##### Assert results of previous test functions. Many wandb functions do not sync to
    ##### api immediately. They **are** synced, however, when the run is finished. The
    ##### results are therefore verified when the run ended.

    # retrieve run
    api._runs = {}  # reset wandb runs internal cache
    run = api.run(path=f"{ENTITY}/{PROJECT}/{run_id}")

    ### assert `test_on_config_update`
    new_config: dict = {}
    dict_merge(new_config, RUN_CONFIG_1)
    dict_merge(new_config, RUN_CONFIG_2)
    stored_config: dict = run.config
    assert (
        stored_config == new_config
    ), "config should have been updated (`test_on_config_update`)"

    ### assert `test_on_metrics`
    history: pd.DataFrame = run.history()
    assert (
        len(history) == 2
    ), "in total 2 metrics should have been uploaded (`test_on_metrics`)"

    ### assert `test_on_table`
    table_name = f"{ENTITY}/{PROJECT}/run-{run_id}-weather:latest"
    table = api.artifact(name=table_name)  # would launch CommError if not existant
    assert table is not None, "table was uploaded as an artifact (`test_on_table`)"

    ### assert `test_on_summary`
    assert "avg_acc" in run.summary
    assert run.summary["avg_acc"] == 0.8

    ###### 🧹 clean up
    run.delete(delete_artifacts=True)
