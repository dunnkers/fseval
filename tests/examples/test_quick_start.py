import subprocess


def install_fseval():
    subprocess.run(
        ["pip", "install", "-e", "."],
        stdout=subprocess.PIPE,
    )


def run_quick_start(cwd):
    install_fseval()
    stdout = subprocess.run(
        [
            "python",
            "benchmark.py",
            "--multirun",
            "ranker=glob(*)",
        ],
        # create database in temporary directory. then check results.
        # "+callbacks.to_sql.url='sqlite:////Users/dunnkers/Downloads/results.sqlite'"
        stdout=subprocess.PIPE,
        cwd=cwd,
    )

    return stdout


def test_quick_start_yaml():
    stdout = run_quick_start("./examples/quick-start-yaml/")

    print("done")
    # TODO check output


def test_basic_structured_config_example():
    stdout = run_quick_start("./examples/quick-start-structured-configs/")

    print("done")
    # TODO check output
