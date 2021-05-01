from tests.test_config import cfg
from fseval.experiment import Experiment
from fseval.types import Ranker


def test_instantiate_experiment(cfg) -> None:
    experiment = Experiment(cfg)
    assert experiment is not None
    assert isinstance(experiment.ranker, Ranker)
