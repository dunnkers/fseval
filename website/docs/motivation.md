---
sidebar_position: 0
---

# Motivation

fseval helps you benchmark **Feature Selection** and **Feature Ranking** algorithms. Any algorithm that ranks features in importance.

It comes useful if you are one of the following types of users:
1. **Feature Selection / Feature Ranker algorithm authors**. You are the author of a novel Feature Selection algorithm. Now, you have to prove the performance of your algorithm against other competitors. Therefore, you are going to run a large-scale benchmark. Many authors, however, spend much time rewriting similar pipelines to benchmark their algorithms. fseval helps you run benchmarks in a structured manner, on supercomputer clusters or on the cloud.
1. **Interpretable AI method authors**. You wrote a new Interpretable AI method that aims to find out which features are most meaningful by ranking them. Now, the challenge is to find out how well your method ranked those features. fseval can help with this.
1. **Machine Learning practitioners**. You have a dataset and want to find out with exactly what features your models will perform best. You can use fseval to try multiple Feature Selection or Feature Ranking algorithms.

## Statement of Need
So why would you need the help of `fseval`? Let's take a look.

Feature Selection (FS) and Feature Ranking (FR) are 
extensively researched topics within machine learning
[@venkatesh2019review; @guyon_introduction_2003]. FS methods determine subsets of relevant
features in a dataset, whereas FR methods rank the features in a dataset
relative to each other in terms of their relevance. When a new FS or FR
method is developed, a benchmarking scheme is necessary to empirically
validate its effectiveness. Often, the benchmark is conducted as
follows: features are ranked by importance, then the predictive quality
of the feature subsets containing the top ranked features is evaluated
using a validation estimator. Some studies let the competing FS or FR
algorithms pick out a fixed number of top $k$ features and validate the
performance of that feature subset
[@roffo_infinite_2015; @zhao_searching_2007; @bradley_feature_1998],
whilst others evaluate multiple subsets of increasing cardinality
containing the highest ranked features
[@wojtas_feature_2020; @bennasar_feature_2015; @gu_generalized_2012; @peng_feature_2005; @kira_feature_1992; @almuallim_learning_1991]. 
FS algorithms that only make a binary prediction on which features to
keep, are always evaluated in the former way.

There is a clear case for performing Feature Selection, as it has been shown to improve classification performance in many tasks, especially those with a large number of features and limited observations. In those applications, it is difficult to determine which FS method is suitable in the general case. Therefore, large empirical comparisons of several FS methods and classifiers are routinely performed. For instance, in microarray data [@cilia2019experimental], medical imaging [@sun2019comparison; @tohka2016comparison; @ashok2016comparison], and text classification [@liu2017multi; @kou2020evaluation]. Therefore, it is valuable to find out emperically which FR- or FS method works best. This requires running a benchmark to do so.

`fseval` is an open-source Python package that helps researchers
perform such benchmarks efficiently by eliminating the need for
implementing benchmarking pipelines from scratch to test new methods.
The pipeline only requires a well-defined configuration file to run -
the rest of the pipeline is automatically executed. Because the entire
experiment setup is deterministic and captured in a configuration file,
results of any experiment can be reproduced given the configuration
file. This can be very convenient to researchers in order to prove the
integrity of their benchmarks.

To the best of our knowledge, there is only one package that aims to accomplish a similar goal (`featsel`, [@reis_featsel_2017]). Compared to this tool, `fseval` is easier to install and use, has better documentation, and is better maintained. `fseval` also has more extensive functionalities compared to `featsel`: with support for easily configurable and reproducible pipeline configuration using either YAML or Python and distributed-processing support. Due to the lack of functionality and the fact that the refered-to library is out-of-date, we consider there to be a gap in the field, which our library aims to fill.

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