from fseval.config import EstimatorConfig, TaskedEstimatorConfig
from fseval.types import Task
from hydra.utils import instantiate
from omegaconf import OmegaConf


def test_estimator_initiation():
    estimator_config = EstimatorConfig(
        estimator={"_target_": "sklearn.tree.DecisionTreeClassifier"}
    )
    estimator_cfg = OmegaConf.create(estimator_config.__dict__)

    tasked_estimator_config = TaskedEstimatorConfig(
        name="some_estimator",
        classifier=estimator_cfg,
        task=Task.classification,
        is_multioutput_dataset=False,
    )
    tasked_estimator_cfg = OmegaConf.create(tasked_estimator_config.__dict__)

    estimator = instantiate(tasked_estimator_cfg)

    assert hasattr(estimator, "estimator")
    assert estimator.name == "some_estimator"
    assert estimator.logger is not None

    estimator.fit([[1, 2]], [0])
    assert hasattr(estimator, "feature_importances_")

    score = estimator.score([[1, 2]], [0])
    assert score >= 0
