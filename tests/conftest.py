from hydra.experimental import initialize, compose
from fseval.config import ExperimentConfig
import pytest


@pytest.fixture(scope="session", autouse=True)
def cfg() -> ExperimentConfig:
    initialize(config_path="../fseval/conf")
    cfg: ExperimentConfig = compose(config_name="config")  # type: ignore
    return cfg
