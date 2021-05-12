from typing import List, cast

import hydra
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf

from fseval.config import BaseConfig
from fseval.pipeline import CallbackList, Pipeline


@hydra.main(config_path="conf", config_name="my_config")
def main(cfg: BaseConfig) -> None:
    primitive_cfg = OmegaConf.to_container(cfg, resolve=True)
    primitive_cfg = cast(dict, primitive_cfg)
    primitive_cfg_pipeline = primitive_cfg.pop("pipeline")

    # take out and instantiate callbacks
    primitive_cfg_callbacks = primitive_cfg.pop("callbacks")
    callbacks = instantiate(primitive_cfg_callbacks)
    callbacks = callbacks.values()

    # put all `pipeline` children into the root config
    pipeline_cfg = OmegaConf.create()
    pipeline_cfg.merge_with(primitive_cfg)
    pipeline_cfg.merge_with(primitive_cfg_pipeline)

    # instantiate pipeline
    pipeline = instantiate(pipeline_cfg)
    pipeline = cast(Pipeline, pipeline)

    # set config and pipeline on callbacks
    callback_list = CallbackList(callbacks)
    callback_list.set_pipeline_config(pipeline.get_config())
    callback_list.set_pipeline(pipeline)

    # run pipeline
    pipeline.run_pipeline(callbacks=callback_list.callbacks)


if __name__ == "__main__":
    main()
