#!/usr/bin/env python

from setuptools import find_namespace_packages, find_packages, setup

with open("README.md", mode="r", encoding="utf8") as fh:
    LONG_DESC = fh.read()
    setup(
        name="fseval",
        version="3.1.0",
        packages=find_namespace_packages(include=["hydra_plugins.*"])
        + find_packages(include=["fseval", "fseval.*"]),
        entry_points={"console_scripts": ["fseval = fseval.main:run_pipeline"]},
        description="Benchmarking framework for Feature Selection and Feature Ranking algorithms 🚀",
        keywords="",
        long_description=LONG_DESC,
        long_description_content_type="text/markdown",
        license="MIT",
        license_file="LICENSE",
        author="Jeroen Overschie",
        author_email="jeroen@darius.nl",
        maintainer="Jeroen Overschie",
        maintainer_email="jeroen@darius.nl",
        url="https://github.com/dunnkers/fseval",
        project_urls={
            "Github": "https://github.com/dunnkers/fseval",
            "Bug Tracker": "https://github.com/dunnkers/fseval/issues",
            "Documentation": "https://dunnkers.com/fseval",
        },
        include_package_data=True,
        install_requires=[
            "hydra-core>=1.1.2",
            "hydra-colorlog>=1.1.0",
            "numpy>=1.19",
            "pandas>=1.1",
            "scikit-learn>=0.24",
            "humanfriendly>=9",
            "shortuuid>=1.0",
            "overrides>=6",
            "SQLAlchemy>=1",
        ],
        python_requires=">= 3.7",
        setup_requires=["black==21.12b0", "pytest-runner>=5"],
        tests_require=["pytest>=6", "pytest-cov>=3", "pytest-dependency"],
        classifiers=[
            "License :: OSI Approved :: MIT License",
            "Development Status :: 4 - Beta",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Operating System :: POSIX :: Linux",
            "Operating System :: MacOS",
            "Operating System :: Microsoft :: Windows",
            "Typing :: Typed",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Software Development",
            "Topic :: Software Development :: Libraries :: Python Modules",
            "Topic :: System :: Benchmark",
        ],
    )
