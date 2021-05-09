from dataclasses import dataclass, field
from typing import Any, Dict

from fseval.base import Configurable, ConfigurableEstimator
from fseval.config import Task
from omegaconf import DictConfig, OmegaConf
from sklearn.base import is_classifier, is_regressor
from sklearn.tree import DecisionTreeClassifier


@dataclass
class SomeValidator(ConfigurableEstimator):
    classifier: Any = DecisionTreeClassifier()
    regressor: Any = None
    task: Task = Task.classification


def test_configurable_estimator():
    clf = SomeValidator()

    # test whether its correctly seen as a classifier
    assert clf._estimator_type == "classifier"
    assert is_classifier(clf)
    assert clf.estimator is not None

    # test config
    config = clf.get_config()
    assert "classifier" not in config
    assert "regressor" not in config
    assert "estimator" in config
    assert (
        config["task"] == "classification"
    ), "`get_config()` should correctly convert an Enum to its `name` value."


@dataclass
class SomeDataset(Configurable):
    feature_importances: DictConfig = field(
        default_factory=lambda: OmegaConf.create({"X[:]": 1.0})
    )


def test_configurable_dataset():
    ds = SomeDataset()

    config = ds.get_config()
    assert config["feature_importances"] == {
        "X[:]": 1.0
    }, """`get_config()` should correctly convert OmegaConf DictConfig object to a 
    regular dict."""
