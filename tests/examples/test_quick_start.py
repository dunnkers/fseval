import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import pytest


def install_fseval():
    subprocess.run(
        ["pip", "install", "-e", "."],
        stdout=subprocess.PIPE,
    )


def run_quick_start(cwd: str, db_url: str):
    subprocess.run(
        [
            "python",
            "benchmark.py",
            "--multirun",
            "ranker=glob(*)",
            f"+callbacks.to_sql.url='{db_url}'",
        ],
        stdout=subprocess.PIPE,
        cwd=cwd,
    )


@pytest.mark.parametrize(
    "cwd",
    ["./examples/quick-start-yaml/", "./examples/quick-start-structured-configs/"],
)
def test_quick_start(cwd: str):
    install_fseval()

    # run pipeline
    db_dir = tempfile.mkdtemp()
    db_file = Path(db_dir) / "results.sqlite"
    db_url = f"sqlite:///{db_file}"
    run_quick_start(cwd=cwd, db_url=db_url)

    # validate scores are in database
    validation_scores = pd.read_sql("validation_scores", con=db_url)
    assert len(validation_scores) == 40
    assert "index" in validation_scores.columns
    assert "n_features_to_select" in validation_scores.columns
    assert "fit_time" in validation_scores.columns
    assert "score" in validation_scores.columns
    assert "bootstrap_state" in validation_scores.columns

    # validate experiment config
    experiments = pd.read_sql("experiments", con=db_url)
    assert len(experiments) == 2
    assert experiments.iloc[0].ranker in ["ANOVA F-value", "Mutual Info"]
    assert experiments.iloc[1].ranker in ["ANOVA F-value", "Mutual Info"]
    assert experiments.iloc[0]["dataset/n"] == 10000
    assert experiments.iloc[0]["dataset/p"] == 20
