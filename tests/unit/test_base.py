from dataclasses import dataclass, field
from typing import Any, Dict

from fseval.base import Configurable
from omegaconf import DictConfig, OmegaConf
from sklearn.tree import DecisionTreeClassifier


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
