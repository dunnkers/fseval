from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class OpenMLDataset:
    """
    Allows loading a dataset from OpenML.

    Attributes:
        dataset_id (int): The dataset ID.
        target_column (str): Which column to use as a target. This column will be used
            as `y`.
        drop_qualitative (bool): Whether to drop any column that is not numeric.
    """

    dataset_id: int = MISSING
    target_column: str = MISSING
    drop_qualitative: bool = False

    # required for instantiation
    _target_: str = "fseval.adapters.openml.OpenML"
