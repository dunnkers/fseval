#!/usr/bin/env python

from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    LONG_DESC = fh.read()
    setup(
        name="fseval",
        version="2.1.2",
        packages=find_packages(include=["fseval", "fseval.*"]),
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
            "hydra-colorlog==1.1.0",
            "numpy>=1.19",
            "pandas>=1.1",
            "scikit-learn>=0.24",
            "humanfriendly>=9.1",
            "shortuuid>=1.0.0",
        ],
        setup_requires=["black==21.4b2", "pytest-runner==5.3.0"],
        tests_require=["pytest==6.2.3"],
        classifiers=[
            "License :: OSI Approved :: MIT License",
            "Development Status :: 4 - Beta",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Operating System :: POSIX :: Linux",
            "Operating System :: MacOS",
            "Operating System :: Microsoft :: Windows",
        ],
    )
