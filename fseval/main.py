from glob import glob
from logging import getLogger
from os import getcwd
from pathlib import Path
from traceback import print_exc
from typing import Dict, Optional, cast

from hydra.core.utils import _save_config
from hydra.utils import instantiate
from omegaconf import DictConfig

from fseval.config import PipelineConfig
from fseval.pipeline.dataset import Dataset, DatasetLoader
from fseval.types import AbstractPipeline, IncompatibilityError, TerminalColor


def run_pipeline(
    cfg: PipelineConfig, raise_incompatibility_errors: bool = False
) -> Optional[Dict]:
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

        # graceful exit message
        exit_strategy: str = (
            "terminating..."
            if raise_incompatibility_errors
            else "exiting gracefully..."
        )
        logger.info(
            "encountered an expected pipeline incompatibility with the current config, "
            + exit_strategy
        )

        # decide between graceful exit or termination
        if raise_incompatibility_errors:
            raise e
        else:
            return None

    # run pipeline
    logger.info(f"starting {TerminalColor.yellow(cfg.pipeline)} pipeline...")
    cfg.storage.load_dir = None  # set these after callbacks were initialized
    cfg.storage.save_dir = None  # set these after callbacks were initialized
    pipeline.callbacks.on_begin(cfg)
    # set storage load- and save dirs
    cfg.storage.load_dir = pipeline.storage.get_load_dir()
    cfg.storage.save_dir = pipeline.storage.get_save_dir()
    pipeline.callbacks.on_config_update(
        {
            "storage": {
                "load_dir": cfg.storage.load_dir,
                "save_dir": cfg.storage.save_dir,
            }
        }
    )
    # save modified cfg to disk
    hydra_output = Path(getcwd()) / ".hydra"
    config = cast(DictConfig, cfg)
    _save_config(config, "config.yaml", hydra_output)
    # log load dir
    logger.info(f"loading files from: {TerminalColor.blue(cfg.storage.load_dir)}")
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

    n_saved_files = len(glob("./*"))
    if cfg.storage.save_dir:
        logger.info(
            f"saved {n_saved_files} files to {TerminalColor.blue(cfg.storage.save_dir)} "
            + TerminalColor.green("✓")
        )
    logger.info(
        f"{TerminalColor.yellow(cfg.pipeline)} pipeline "
        + f"finished {TerminalColor.green('✓')}"
    )
    pipeline.callbacks.on_summary(scores)
    pipeline.callbacks.on_end()

    # return final scores
    scores = cast(Dict, scores)
    return scores
