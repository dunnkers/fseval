#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="fseval",
    version="2.0",
    packages=find_packages(include=["fseval"]),
    entry_points={"console_scripts": ["fseval = fseval.main:main"]},
    author="Jeroen Overschie",
    url="https://dunnkers.com/",
    include_package_data=True,
    install_requires=[
        "hydra-core==1.1.0.dev6",
        "-e git+https://github.com/dunnkers/hydra.git@fe322a8f95eb6badcdce001375c727567545e97b#egg=hydra_rq_launcher&subdirectory=plugins/hydra_rq_launcher",
        "numpy==1.20.2",
        "openml==0.12.1",
        "pandas==1.2.4",
        "scikit-learn==0.24.2",
        "wandb==0.10.28",
    ],
)
