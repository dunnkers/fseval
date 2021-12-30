#!/usr/bin/env python

from setuptools import find_namespace_packages, find_packages, setup

with open("README.md", "r") as fh:
    LONG_DESC = fh.read()
    setup(
        name="fseval",
        version="2.1.2",
        packages=find_namespace_packages(include=["hydra_plugins.*"])
        + find_packages(include=["fseval", "fseval.*"]),
        entry_points={"console_scripts": ["fseval = fseval.main:main"]},
        description="Benchmarking framework for Feature Selection algorithms 🚀",
        long_description=LONG_DESC,
        long_description_content_type="text/markdown",
        license="MIT",
        author="Jeroen Overschie",
        url="https://github.com/dunnkers/fseval",
        include_package_data=True,
        install_requires=[
            "hydra-core>=1.1.0",
            "hydra-colorlog>=1.1.0",
            "numpy>=1.19",
            "pandas>=1.1",
            "scikit-learn>=0.24",
            "humanfriendly>=9",
            "shortuuid>=1.0",
        ],
        setup_requires=["black==21.12b0", "pytest-runner>=5"],
        tests_require=["pytest>=6", "pytest-cov>=3", "pytest-dependency"],
        classifiers=[
            "License :: OSI Approved :: MIT License",
            "Development Status :: 4 - Beta",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Operating System :: POSIX :: Linux",
            "Operating System :: MacOS",
            "Operating System :: Microsoft :: Windows",
        ],
    )
