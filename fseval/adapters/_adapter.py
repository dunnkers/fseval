from typing import Tuple, List, Optional
from dataclasses import dataclass
from fseval.base import Configurable


@dataclass
class Adapter(Configurable):
    _target_: Optional[str] = None

    def get_data(self) -> Tuple[List, List]:
        raise NotImplementedError
