from typing import List, cast

import hydra
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.pipeline import Pipeline

from fseval.callbacks._callback import CallbackList
from fseval.config import BaseConfig
from fseval.cv import CrossValidator
from fseval.datasets import Dataset


@hydra.main(config_path="conf", config_name="my_config")
def main(cfg: BaseConfig) -> None:
    dataset: Dataset = instantiate(cfg.dataset)  # pipeline.
    cv: CrossValidator = instantiate(cfg.cv)  # pipeline.
    callbacks = instantiate(cfg.callbacks)
    callback_list = CallbackList(callbacks.values())

    dataset.load()
    X, y = dataset.X, dataset.y
    X_train, X_test, y_train, y_test = cv.train_test_split(X, y)

    pipeline: Pipeline = instantiate(cfg.pipeline, callback_list=callback_list)
    pipeline.fit(X_train, y_train)
    pipeline.score(X_test, y_test)

    # primitive_cfg = OmegaConf.to_container(cfg, resolve=True)
    # primitive_cfg = cast(dict, primitive_cfg)

    # # instantiate callbacks
    # callbacks = instantiate(cfg.callbacks)
    # callbacks = callbacks.values()
    # callback_list = CallbackList(callbacks)
    # callback_list.set_pipeline_config(primitive_cfg)

    # # take out pipeline and callbacks
    # primitive_cfg_pipeline = primitive_cfg.pop("pipeline")
    # primitive_cfg.pop("callbacks")

    # # put all `pipeline` children into the root config
    # pipeline_cfg = OmegaConf.create()
    # pipeline_cfg.merge_with(primitive_cfg)
    # pipeline_cfg.merge_with(primitive_cfg_pipeline)

    # # instantiate pipeline
    # pipeline = instantiate(pipeline_cfg, callback_list=callback_list)
    # pipeline = cast(Pipeline, pipeline)

    # # run pipeline
    # pipeline.run_pipeline()


if __name__ == "__main__":
    main()
