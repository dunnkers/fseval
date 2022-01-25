from fseval.config import DatasetConfig
from fseval.types import Task

synthetic_dataset = DatasetConfig(
    name="My synthetic dataset",
    task=Task.classification,
    adapter=dict(
        _target_="sklearn.datasets.make_classification",
        n_samples=10000,
        n_informative=2,
        n_classes=2,
        n_features=20,
        n_redundant=0,
        random_state=0,
        shuffle=False,
    ),
    feature_importances={"X[:, 0:2]": 1.0},
)
