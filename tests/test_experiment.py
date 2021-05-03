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


# def test_experiment_run(cfg) -> None:
#     experiment = Experiment(cfg)
# experiment.run()
