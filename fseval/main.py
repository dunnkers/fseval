from copy import deepcopy
from logging import getLogger
from traceback import print_exc
from typing import Dict, cast

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fseval.config import BaseConfig
from fseval.pipeline.cv import CrossValidator
from fseval.pipeline.dataset import Dataset, DatasetLoader
from fseval.pipelines._callback_collection import CallbackCollection
from fseval.types import (
    AbstractPipeline,
    AbstractStorageProvider,
    Callback,
    IncompatibilityError,
    TerminalColor,
)


@hydra.main(config_path="conf", config_name="my_config")
def main(cfg: DictConfig) -> None:
    logger = getLogger(__name__)
    logger.info("instantiating pipeline components...")

    # instantiate and load dataset
    dataset_loader: DatasetLoader = instantiate(cfg.dataset)
    dataset: Dataset = dataset_loader.load()
    # set 'runtime' properties. the other pipeline components need them.
    cfg.dataset.n = dataset.n
    cfg.dataset.p = dataset.p
    cfg.dataset.multioutput = dataset.multioutput

    # instantiate pipeline
    logger.info(f"instantiating pipeline...")
    try:
        pipeline: AbstractPipeline = instantiate(cfg)
    except IncompatibilityError as e:
        (msg,) = e.args
        logger.error(msg)
        logger.info(
            "encountered an expected pipeline incompatibility with the current config, "
            + "exiting gracefully..."
        )
        return

    # run pipeline
    logger.info(f"starting {cfg.pipeline} pipeline...")
    pipeline.callbacks.on_begin(cfg)
    # load dataset and cv split
    logger.info(
        f"using dataset: {TerminalColor.yellow(dataset_loader.name)} "
        + f"[{dataset._log_details}]"
    )
    X, y = dataset.X, dataset.y
    X_train, X_test, y_train, y_test = pipeline.cv.train_test_split(X, y)

    try:
        logger.info(f"pipeline {TerminalColor.cyan('prefit')}...")
        pipeline.prefit()
        logger.info(f"pipeline {TerminalColor.cyan('fit')}...")
        pipeline.fit(X_train, y_train)
        logger.info(f"pipeline {TerminalColor.cyan('postfit')}...")
        pipeline.postfit()
        logger.info(f"pipeline {TerminalColor.cyan('score')}...")
        scores = pipeline.score(
            X_test, y_test, feature_importances=dataset.feature_importances
        )
    except Exception as e:
        print_exc()
        logger.error(e)
        logger.info(
            "error occured during pipeline `prefit`, `fit`, `postfit` or `score` step... "
            + "exiting with a status code 1."
        )
        pipeline.callbacks.on_end(exit_code=1)
        raise e

    logger.info(f"{cfg.pipeline} pipeline finished {TerminalColor.green('✓')}")
    pipeline.callbacks.on_summary(scores)
    pipeline.callbacks.on_end()


if __name__ == "__main__":
    main()
