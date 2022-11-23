---
sidebar_position: 0
---

# Motivation

`fseval` helps you benchmark **Feature Selection** and **Feature Ranking** algorithms. Any algorithm that ranks features in importance.

It comes useful if you are one of the following types of users:
1. **Feature Selection / Feature Ranker algorithm authors**. You are the author of a novel Feature Selection algorithm. Now, you have to prove the performance of your algorithm against other competitors. Therefore, you are going to run a large-scale benchmark. Many authors, however, spend much time rewriting similar pipelines to benchmark their algorithms. fseval helps you run benchmarks in a structured manner, on supercomputer clusters or on the cloud.
1. **Interpretable AI method authors**. You wrote a new Interpretable AI method that aims to find out which features are most meaningful by ranking them. Now, the challenge is to find out how well your method ranked those features. fseval can help with this.
1. **Machine Learning practitioners**. You have a dataset and want to find out with exactly what features your models will perform best. You can use fseval to try multiple Feature Selection or Feature Ranking algorithms.

## Statement of Need

<details>
<summary>
About benchmarking Feature Selection and Feature Ranking algorithms
</summary>

Feature Selection (FS) and Feature Ranking (FR) are extensively researched topics within machine learning ([Venkatesh et al, 2019](https://sciendo.com/it/article/10.2478/cait-2019-0001), [Guyon et al, 2003](https://www.jmlr.org/papers/volume3/guyon03a/guyon03a.pdf?ref=driverlayer.com/web)). FS methods determine subsets of relevant features in a dataset, whereas FR methods rank the features in a dataset relative to each other in terms of their relevance. When a new FS or FR method is developed, a benchmarking scheme is necessary to empirically validate its effectiveness. Often, the benchmark is conducted as follows: features are ranked by importance, then the predictive quality of the feature subsets containing the top ranked features is evaluated using a validation estimator. Some studies let the competing FS or FR algorithms pick out a fixed number of top `k` features and validate the performance of that feature subset ([Roffo et al, 2015](http://ieeexplore.ieee.org/document/7410835/), [Zhao et al, 2007](https://www.semanticscholar.org/paper/Searching-for-Interacting-Features-Zhao-Liu/d2debe138a9b67d838b11d622651383322934aee), [Bradley et al, 1998](http://www.machine-learning.martinsewell.com/feature-selection/BradleyMangasarian1998.pdf)), whilst others evaluate multiple subsets of increasing cardinality containing the highest ranked features ([Wojtas et al, 2022](http://arxiv.org/abs/2010.08973), [Bennasar et al, 2015](http://www.sciencedirect.com/science/article/pii/S0957417415004674), [Gu et al, 2012](http://arxiv.org/abs/1202.3725), [Peng et al, 2005](https://ieeexplore.ieee.org/abstract/document/1453511), [Kira et al, 2005](https://www.aaai.org/Library/AAAI/1992/aaai92-020.php), [Almuallim et al, 1991](https://web.engr.oregonstate.edu/~tgd/publications/aaai91-focus.ps.gz)).  FS algorithms that only make a binary prediction on which features to keep, are always evaluated in the former way.
</details>


There is a clear case for performing Feature Selection, as it has been shown to improve classification performance in many tasks, especially those with a large number of features and limited observations. In those applications, it is difficult to determine which FS method is suitable in the general case. Therefore, large empirical comparisons of several FS methods and classifiers are routinely performed. For instance, in microarray data ([Cilia et al, 2019](https://www.mdpi.com/2078-2489/10/3/109)), medical imaging ([Sun et al, 2019](https://ieeexplore.ieee.org/abstract/document/8763934/), [Tohka et al, 2016](https://link.springer.com/article/10.1007/s12021-015-9292-3), [Ashok et al, 2016](https://d1wqtxts1xzle7.cloudfront.net/47557926/L601019499-with-cover-page-v2.pdf?Expires=1668764412&Signature=GnBEHq3XrO1yvRbtiEPcnxb3WEXlpA99mgUICAngTrijKwEFt9l2SDZgj7sZmOn1HVsO6wX2gfEHmI7VDjBOQgkUcrviNCfE432Iu2VxQ2BsI0LN~NR29FI8v-dvFCJPsDHEBuN1Sgr48d4rxc-QiJSOXCYZJ-nYQzbBEs~VxVJLvQnkrpeIcS7HN3NN-EaH4Kx~DviXAQSIgWEWuNfLyQmWDaQh8gAIDCk916wLP8Eri-s53Q3L2GQU1mwLqUF9ZMBmMaFtw6hbcADoi7cHqLiafQU5HADqUyawNUAWbBTf~qonfgn1rsj3f2FNkN3Nn~yO9ihG35VlwGWYmRvirA__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)), and text classification ([Liu et al, 2017](https://www.sciencedirect.com/science/article/abs/pii/S0957417417301951), [Kou et al, 2020](https://www.sciencedirect.com/science/article/pii/S1568494619306179)). Therefore, it is valuable to find out emperically which FR- or FS method works best. This requires running a benchmark to do so.

`fseval` is an open-source Python package that helps researchers
perform such benchmarks efficiently by eliminating the need for
implementing benchmarking pipelines from scratch to test new methods.
The pipeline only requires a well-defined configuration file to run -
the rest of the pipeline is automatically executed. Because the entire
experiment setup is deterministic and captured in a configuration file,
results of any experiment can be reproduced given the configuration
file. This can be very convenient to researchers in order to prove the
integrity of their benchmarks.

- The **target audiences** are researchers in the domains of Feature Selection and Feature Ranking, as well as businesses that are looking for the best FR- or FS method to use for their use case.
- The **scope** of `fseval` is limited to
handle tabular datasets for the classification and regression
objectives.

## Key features 🚀
Most importantly, `fseval` has the following in store for you.

- Easily benchmark Feature Ranking algorithms
- Built on [Hydra](https://hydra.cc/)
- Support for distributed systems (SLURM through the [Submitit launcher](https://hydra.cc/docs/plugins/submitit_launcher), AWS support through the [Ray launcher](https://hydra.cc/docs/plugins/ray_launcher/))
- Reproducible experiments (your entire experiment can be described and reproduced by 1 YAML file)
- Send experiment results directly to a dashboard (integration with [Weights and Biases](https://wandb.ai/) is built-in)
- Export your data to any SQL database (integration with [SQLAlchemy](https://www.sqlalchemy.org/))