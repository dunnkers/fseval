# Analyze algorithm stability

For many applications, it is very important the algorithms that are used are **stable** enough. This means, that when a different sample of data is taken from some distribution, the results will turn out similar. This, combined with possible inherent stochastic properties of an algorithm, make up for the _stability_ of the algorithm. The same applies to Feature Selection or Feature Ranking algorithms.

Therefore, let's do such an experiment! We are going to compare the stability of [ReliefF](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.56.4740&rep=rep1&type=pdf) to [Boruta](https://www.jstatsoft.org/article/view/v036i11), two popular feature selection algorithms. We are going to do this using a metric introduced in [Nogueira et al, 2018](https://www.jmlr.org/papers/volume18/17-514/17-514.pdf).


## The experiment

We are going to run an experiment with the following [configuration](https://github.com/dunnkers/fseval/tree/master/examples/algorithm-stability-yaml/). 

Download the experiment config: [algorithm-stability-yaml.zip](pathname:///fseval/zipped-examples/algorithm-stability-yaml.zip)

Most notably are the following configuration settings:

```yaml title="my_config.yaml"
defaults:
  - base_pipeline_config
  - _self_
  - override dataset: synclf_hard
  - override validator: knn
  - override /callbacks:
      - to_sql
  // highlight-start
  - override /metrics:
      - stability_nogueira
  // highlight-end

// highlight-start
n_bootstraps: 10
// highlight-end
```

That means, we are going to generate a synthetic dataset and sample 10 subsets from it. This is because `n_bootstraps=10`. Then, after the feature selection algorithm was executed and fitted on the dataset, a custom installed metric will be executed, called `stability_nogueira`. This can be found in the `/conf/metrics` folder, which in turn refers to a class in the `benchmark.py` file.

To now run the experiment, run the following command inside the `algorithm-stability-yaml` folder:

```shell
python benchmark.py --multirun ranker="glob(*)" +callbacks.to_sql.url="sqlite:///$HOME/results.sqlite"
```

## Analyzing the results

### Recap

Hi! Let's analyze the results of the experiment you just ran. To **recap**:

1. You just ran something similar to:

    `python benchmark.py --multirun ranker="glob(*)" +callbacks.to_sql.url="sqlite:///$HOME/results.sqlite"`
2. There now should exist a `.sqlite` file at this path: `$HOME/results.sqlite`:

    ```
    $ ls -al $HOME/results.sqlite
    -rw-r--r-- 1 vscode vscode 20480 Sep 21 08:16 /home/vscode/results.sqlite
    ```

Let's now analyze the results! 📈

### Analysis

> The rest of the text assumes all code was ran inside a Jupyter Notebook, in chronological order. The source Notebook can be found [here](https://github.com/dunnkers/fseval/tree/master/examples/algorithm-stability-yaml/analyze-results.ipynb)

First, we will install `plotly-express`, so we can make nice plots later.


```python
%pip install plotly-express --quiet
```


Figure out the SQL connection URI.


```python
import os

con: str = "sqlite:///" + os.environ["HOME"] + "/results.sqlite"
con
```




    'sqlite:////home/vscode/results.sqlite'



Read in the `experiments` table. This table contains metadata for all 'experiments' that have been run.


```python
import pandas as pd

experiments: pd.DataFrame = pd.read_sql_table("experiments", con=con, index_col="id")
experiments
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr >
      <th></th>
      <th>dataset</th>
      <th>dataset/n</th>
      <th>dataset/p</th>
      <th>dataset/task</th>
      <th>dataset/group</th>
      <th>dataset/domain</th>
      <th>ranker</th>
      <th>validator</th>
      <th>local_dir</th>
      <th>date_created</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>38vqcwus</th>
      <td>Synclf hard</td>
      <td>10000</td>
      <td>50</td>
      <td>classification</td>
      <td>Synclf</td>
      <td>synthetic</td>
      <td>Boruta</td>
      <td>k-NN</td>
      <td>/workspaces/fseval/examples/algorithm-stabilit...</td>
      <td>2022-09-21 08:22:28.965510</td>
    </tr>
    <tr>
      <th>y6bb1hcc</th>
      <td>Synclf hard</td>
      <td>1000</td>
      <td>50</td>
      <td>classification</td>
      <td>Synclf</td>
      <td>synthetic</td>
      <td>Boruta</td>
      <td>k-NN</td>
      <td>/workspaces/fseval/examples/algorithm-stabilit...</td>
      <td>2022-09-21 08:22:53.609396</td>
    </tr>
    <tr>
      <th>3vtr13pg</th>
      <td>Synclf hard</td>
      <td>1000</td>
      <td>50</td>
      <td>classification</td>
      <td>Synclf</td>
      <td>synthetic</td>
      <td>ReliefF</td>
      <td>k-NN</td>
      <td>/workspaces/fseval/examples/algorithm-stabilit...</td>
      <td>2022-09-21 08:25:09.974370</td>
    </tr>
  </tbody>
</table>
</div>



That's looking good 🙌🏻.

Now, let's read in the `stability` table. We put data in this table by using our custom-made metric, defined in the `StabilityNogueira` class in `benchmark.py`. There, we push data to this table using `callbacks.on_table`.


```python
stability: pd.DataFrame = pd.read_sql_table("stability", con=con, index_col="id")
stability
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr >
      <th></th>
      <th>index</th>
      <th>stability</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>y6bb1hcc</th>
      <td>0</td>
      <td>0.933546</td>
    </tr>
    <tr>
      <th>3vtr13pg</th>
      <td>0</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Cool. Now let's join the experiments with their actual metrics.


```python
stability_experiments = stability.join(experiments)
stability_experiments
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr >
      <th></th>
      <th>index</th>
      <th>stability</th>
      <th>dataset</th>
      <th>dataset/n</th>
      <th>dataset/p</th>
      <th>dataset/task</th>
      <th>dataset/group</th>
      <th>dataset/domain</th>
      <th>ranker</th>
      <th>validator</th>
      <th>local_dir</th>
      <th>date_created</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>y6bb1hcc</th>
      <td>0</td>
      <td>0.933546</td>
      <td>Synclf hard</td>
      <td>1000</td>
      <td>50</td>
      <td>classification</td>
      <td>Synclf</td>
      <td>synthetic</td>
      <td>Boruta</td>
      <td>k-NN</td>
      <td>/workspaces/fseval/examples/algorithm-stabilit...</td>
      <td>2022-09-21 08:22:53.609396</td>
    </tr>
    <tr>
      <th>3vtr13pg</th>
      <td>0</td>
      <td>1.000000</td>
      <td>Synclf hard</td>
      <td>1000</td>
      <td>50</td>
      <td>classification</td>
      <td>Synclf</td>
      <td>synthetic</td>
      <td>ReliefF</td>
      <td>k-NN</td>
      <td>/workspaces/fseval/examples/algorithm-stabilit...</td>
      <td>2022-09-21 08:25:09.974370</td>
    </tr>
  </tbody>
</table>
</div>



Finally, we can plot the results so we can get a better grasp of what's going on:


```python
import plotly.express as px

px.bar(stability_experiments,
    x="ranker",
    y="stability"
)
```


![feature selectors algorithm stability](/img/recipes/feature-selectors-stability-barplot.png)


We can now observe that for Boruta and ReliefF, ReliefF is the most 'stable' given this dataset, getting 100% the same features for all 10 bootstraps that were run.
