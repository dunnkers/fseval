from dataclasses import dataclass
from typing import List, Optional, Tuple

from sklearn.base import BaseEstimator


@dataclass
class Adapter(BaseEstimator):
    _target_: Optional[str] = None

    def get_data(self) -> Tuple[List, List]:
        raise NotImplementedError
