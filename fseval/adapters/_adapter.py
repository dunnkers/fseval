from dataclasses import dataclass
from typing import List, Optional, Tuple

from fseval.base import Configurable


@dataclass
class Adapter(Configurable):
    _target_: Optional[str] = None

    def get_data(self) -> Tuple[List, List]:
        raise NotImplementedError
