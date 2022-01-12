import os
import time
from logging import Logger, getLogger
from pathlib import Path

import pandas as pd
from fseval.types import TerminalColor
from omegaconf import DictConfig, OmegaConf

from .to_csv import CSVCallback


class ExcelCallback(CSVCallback):
    """Excel support for fseval. Writes tables to a single excel file as worksheets.
    The experiment config is written to a `experiments` worksheet.
    """

    ...
