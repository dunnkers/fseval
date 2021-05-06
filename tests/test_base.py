from dataclasses import dataclass
from typing import Any

from sklearn.base import is_classifier, is_regressor
from sklearn.tree import DecisionTreeClassifier

from fseval.base import ConfigurableEstimator
from fseval.config import Task


@dataclass
class SomeValidator(ConfigurableEstimator):
    classifier: Any = DecisionTreeClassifier()
    regressor: Any = None
    task: Task = Task.classification


def test_configurable_estimator():
    clf = SomeValidator()
    assert clf._estimator_type == "classifier"
    assert is_classifier(clf)
    assert clf.estimator is not None
    assert "classifier" not in clf.get_config()
    assert "regressor" not in clf.get_config()
    assert "estimator" in clf.get_config()
