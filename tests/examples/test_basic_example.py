import subprocess

import pytest


@pytest.mark.skip(reason="WIP")
def test_basic_example():
    useless_cat_call = subprocess.run(
        ["python", "examples/comparing-two-feature-selectors/benchmark.py"],
        stdout=subprocess.PIPE,
    )
    print(useless_cat_call.stdout)  # Hello from the other side
