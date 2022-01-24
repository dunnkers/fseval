from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING


@dataclass
class StorageConfig:
    """
    Allows you to define a storage for loading and saving cached estimators, among other
    files, like the hydra and fseval configuration in YAML.

    Attributes:
        load_dir (Optional[str]): Defines a path to load files from. Must point to
            exactly the directory containing the files, i.e. you should not point to a
            higher-level directory than where the files are. Path can be relative or
            absolute, but an absolute path is recommended.
        save_dir (Optional[str]): The directory to save files to. Can be relative or
            absolute.
    """

    load_dir: Optional[str] = None
    save_dir: Optional[str] = None

    # required for instantiation
    _target_: str = MISSING


@dataclass
class MockStorageConfig(StorageConfig):
    """
    Disables storage.

    Attributes:
        load_dir (str): The directory to load files from
        save_dir (str): The directory to save files to
    """

    # required for instantiation
    _target_: str = "fseval.storage.mock.MockStorage"


@dataclass
class LocalStorageConfig(MockStorageConfig):
    """
    Saves files to a local directory.

    Attributes:
        load_dir (str): The directory to load files from
        save_dir (str): The directory to save files to
    """

    # required for instantiation
    _target_: str = "fseval.storage.local.LocalStorage"


@dataclass
class WandbStorageConfig(LocalStorageConfig):
    """
    Storage for Weights and Biases (wandb), allowing users to save- and
    restore files to the service.

    Attributes:
        load_dir (Optional[str]): when set, an attempt is made to load from the
            designated local directory first, before downloading the data off of wandb.
            Can be used to perform faster loads or prevent being rate-limited on wandb.
        save_dir (Optional[str]): when set, uses this directory to save files, instead
            of the usual wandb run directory, under the `files` subdirectory.
        entity (Optional[str]): allows you to recover from a specific entity,
            instead of using the entity that is set for the 'current' run.
        project (Optional[str]): recover from a specific project.
        run_id (Optional[str]): recover from a specific run id.
        save_policy (str): policy for `wandb.save`. Can be 'live', 'now' or 'end'.
            Determines at which point of the run the file is uploaded. Defaults to
            "live".
    """

    entity: Optional[str] = None
    project: Optional[str] = None
    run_id: Optional[str] = None
    save_policy: Optional[str] = "live"

    # required for instantiation
    _target_: str = "fseval.storage.wandb.WandbStorage"
