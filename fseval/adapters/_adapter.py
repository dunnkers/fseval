from typing import Tuple, List


class Adapter:
    def get_data(self) -> Tuple[List, List]:
        raise NotImplementedError
