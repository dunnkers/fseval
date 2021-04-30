from dataclasses import dataclass
from openml.datasets import get_dataset
import pandas as pd
import numpy as np
from typing import Tuple, List
import wandb


@dataclass
class DataSource:
    """
    Args:
        type (str): Either 'classification' or 'regression'.
        name (str): A human-friendly name for this data source.
    """

    name: str
    type: str
    multivariate: bool = False
    relevant_features: List[int] = None

    def load(self) -> Tuple[List, List]:
        raise NotImplementedError

    def get_data(self) -> Tuple[List, List]:
        X, y = self.load()

        # samples / dimensions
        n, p = np.shape(X)
        self.n = n
        self.p = p

        return X, y


@dataclass
class OpenML(DataSource):
    id: int = None
    target_column: str = None

    def load(self) -> Tuple[List, List]:
        dataset = get_dataset(self.id)
        X, y, cat, _ = dataset.get_data(target=self.target_column)

        # drop qualitative columns
        to_drop = X.columns[np.array(cat)]
        X = X.drop(columns=to_drop).values

        # quantitatively encode target
        y, _ = pd.factorize(y)

        return X, y


@dataclass
class WandbArtifact(DataSource):
    artifact_id: str = None
    artifact_type: str = "dataset"
    entity: str = None

    def load(self) -> Tuple[List, List]:
        api = wandb.Api()
        artifact = api.artifact(self.artifact_id)
        X = artifact.get("X").data
        Y = artifact.get("Y").data
        return np.array(X), np.array(Y)
