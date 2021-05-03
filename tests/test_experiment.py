from fseval.experiment import Experiment
from fseval.rankers import Ranker


def test_instantiate_experiment(cfg) -> None:
    experiment = Experiment(cfg)
    assert experiment is not None
    assert isinstance(experiment.ranker, Ranker)


def test_experiment_run(cfg) -> None:
    experiment = Experiment(cfg)
    experiment.run()
