from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class WandbDataset:
    """
    Loads a dataset from the Weights and Biases artifact store. Requires being logged
    into the Weights and Biases CLI (in other words, having the `WANDB_API_KEY` set),
    and having installed the `wandb` python package.

    Attributes:
        artifact_id (str): The ID of the artifact to fetch. Has to be of the following
            form: `<entity>/<project>/<artifact_name>:<artifact_version>`. For example:
            `dunnkers/synthetic-datasets/switch:v0` would be a valid artifact_id.
    """

    artifact_id: str = MISSING

    # required for instantiation
    _target_: str = "fseval.adapters.wandb.Wandb"
