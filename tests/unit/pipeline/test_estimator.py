from hydra.utils import instantiate
from omegaconf import OmegaConf

from fseval.types import Task


def test_estimator_initiation():
    estimator_cfg = OmegaConf.create(
        {"estimator": {"_target_": "sklearn.tree.DecisionTreeClassifier"}}
    )
    cfg = OmegaConf.create(
        {
            "_target_": "fseval.pipeline.estimator.instantiate_estimator",
            "_recursive_": False,
            "_target_class_": "fseval.pipeline.estimator.Estimator",
            "name": "some_estimator",
            "task": Task.classification,
            "classifier": estimator_cfg,
            "regressor": None,
        }
    )
    estimator = instantiate(cfg)

    assert hasattr(estimator, "estimator")
    assert estimator.name == "some_estimator"
    assert estimator.logger is not None

    estimator.fit([[1, 2]], [0])
    assert hasattr(estimator, "feature_importances_")

    score = estimator.score([[1, 2]], [0])
    assert score >= 0
