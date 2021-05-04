from fseval.experiment import Experiment
from fseval.rankers import Ranker
from hydra.utils import instantiate


def test_instantiate_experiment(cfg) -> None:
    experiment = instantiate(cfg)
    assert experiment is not None
    assert isinstance(experiment.ranker, Ranker)


def test_experiment_params(cfg) -> None:
    experiment = instantiate(cfg)
    params = experiment.get_params()
    assert params["dataset"] is not None
    assert params["cv__fold"] == 0


def test_experiment_config(cfg) -> None:
    experiment = instantiate(cfg)
    config = experiment.get_config()
    assert config["dataset"] is not None
    assert isinstance(config["ranker"]["estimator"], dict)
    assert isinstance(config["dataset"]["adapter"], dict)
