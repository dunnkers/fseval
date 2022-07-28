---
sidebar_position: 0
---

# Motivation

fseval helps you benchmark **Feature Selection** and **Feature Ranking** algorithms. Any algorithm that ranks features in importance.

It comes useful if you are one of the following types of users:
1. **Feature Selection / Feature Ranker algorithm authors**. You are the author of a novel Feature Selection algorithm. Now, you have to prove the performance of your algorithm against other competitors. Therefore, you are going to run a large-scale benchmark. Many authors, however, spend much time rewriting similar pipelines to benchmark their algorithms. fseval helps you run benchmarks in a structured manner, on supercomputer clusters or on the cloud.
1. **Interpretable AI method authors**. You wrote a new Interpretable AI method that aims to find out which features are most meaningful by ranking them. Now, the challenge is to find out how well your method ranked those features. fseval can help with this.
1. **Machine Learning practitioners**. You have a dataset and want to find out with exactly what features your models will perform best. You can use fseval to try multiple Feature Selection or Feature Ranking algorithms.



Key features 🚀:
- Easily benchmark Feature Ranking algorithms
- Built on [Hydra](https://hydra.cc/)
- Support for distributed systems (SLURM through the [Submitit launcher](https://hydra.cc/docs/plugins/submitit_launcher), AWS support through the [Ray launcher](https://hydra.cc/docs/plugins/ray_launcher/))
- Reproducible experiments (your entire experiment can be described and reproduced by 1 YAML file)
- Send experiment results directly to a dashboard (integration with [Weights and Biases](https://wandb.ai/) is built-in)
- Export your data to any SQL database (integration with [SQLAlchemy](https://www.sqlalchemy.org/))