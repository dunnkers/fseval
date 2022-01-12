import subprocess


def test_basic_example():
    subprocess.run(
        ["pip", "install", "-e", "."],
        stdout=subprocess.PIPE,
    )
    subprocess.run(
        ["pip", "install", "openml"],
        stdout=subprocess.PIPE,
    )
    benchmark_stdout = subprocess.run(
        [
            "python",
            "benchmark.py",
            "validator=knn",
            "dataset=glob(*)",
            "ranker=glob(*)",
            "--multirun",
        ],
        stdout=subprocess.PIPE,
        cwd="./examples/basic-example/",
    )

    print("done")
    # TODO check output


def test_basic_structured_config_example():
    subprocess.run(
        ["pip", "install", "-e", "."],
        stdout=subprocess.PIPE,
    )
    subprocess.run(
        ["pip", "install", "openml"],
        stdout=subprocess.PIPE,
    )
    benchmark_stdout = subprocess.run(
        [
            "python",
            "benchmark.py",
            "validator=knn",
            "dataset=glob(*)",
            "ranker=glob(*)",
            "--multirun",
        ],
        stdout=subprocess.PIPE,
        cwd="./examples/basic-example-structured-configs/",
    )

    print("done")
    # TODO check output
