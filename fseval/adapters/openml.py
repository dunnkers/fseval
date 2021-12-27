from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from fseval.types import AbstractAdapter
from omegaconf import MISSING

from openml.datasets import get_dataset


@dataclass
class OpenMLDataset:
    _target_: str = "fseval.adapters.openml.OpenML"
    dataset_id: int = MISSING
    target_column: str = MISSING


@dataclass
class OpenML(AbstractAdapter, OpenMLDataset):
    def get_data(self) -> Tuple[List, List]:
        dataset = get_dataset(self.dataset_id)
        X, y, cat, _ = dataset.get_data(target=self.target_column)

        # drop qualitative columns
        to_drop = X.columns[np.array(cat)]
        X = X.drop(columns=to_drop).values

        # quantitatively encode target
        y, _ = pd.factorize(y)

        return X, y
