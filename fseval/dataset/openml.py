from dataclasses import dataclass
from fseval.config import DatasetConfig
from typing import Tuple, List
from openml.datasets import get_dataset
import numpy as np
import pandas as pd


@dataclass
class OpenML(DatasetConfig):
    def load(self) -> Tuple[List, List]:
        dataset = get_dataset(self.identifier)
        X, y, cat, _ = dataset.get_data(target=self.misc.target_column)

        # drop qualitative columns
        to_drop = X.columns[np.array(cat)]
        X = X.drop(columns=to_drop).values

        # quantitatively encode target
        y, _ = pd.factorize(y)

        return X, y
