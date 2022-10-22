# Comparing Feature Selectors
Hi! You want to compare the performance of multiple feature selectors? This is an example Notebook, showing you how to do such an analysis. 

## Prerequisites

We are going to use more or less the same configuration as we did in the [Quick start](../../quick-start) example, but then with more Feature Selectors. Again, start by downloading the example project: [comparing-feature-selectors.zip](pathname:///fseval/zipped-examples/comparing-feature-selectors.zip)

### Installing the required packages

Now, let's install the required packages. Make sure you are in the `comparing-feature-selectors` folder, containing the `requirements.txt` file, and then run the following:

```
pip install -r requirements.txt
```

## Running the experiment

Run the following command to start the experiment:

```
python benchmark.py --multirun ranker="glob(*)" +callbacks.to_sql.url="sqlite:////tmp/results.sqlite
```

## Analyzing the results

There now should exist a `.sqlite` file at this path: `/tmp/results.sqlite`:

    ```
    $ ls -al /tmp/results.sqlite
    -rw-r--r-- 1 vscode vscode 20480 Sep 21 08:16 /tmp/results.sqlite
    ```

Is that the case? Then let's now analyze the results! 📈

We will install `plotly-express`, so we can make nice plots later.


```python
%pip install plotly-express nbconvert --quiet
```


Next, let's find a place to store our results to. In this case, we choose to store it in a local SQLite database, located at `/tmp/results.sqlite`.


```python
import os

con: str = "sqlite:////tmp/results.sqlite"
con
```




    'sqlite:////tmp/results.sqlite'



Now, we can read the `experiments` table.


```python
import pandas as pd

experiments: pd.DataFrame = pd.read_sql_table("experiments", con=con, index_col="id")
experiments
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
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
      <th>3lllxl48</th>
      <td>My synthetic dataset</td>
      <td>10000</td>
      <td>20</td>
      <td>classification</td>
      <td>None</td>
      <td>None</td>
      <td>ANOVA F-value</td>
      <td>k-NN</td>
      <td>/workspaces/fseval/examples/comparing-feature-...</td>
      <td>2022-10-22 14:28:27.506838</td>
    </tr>
    <tr>
      <th>1944ropg</th>
      <td>My synthetic dataset</td>
      <td>10000</td>
      <td>20</td>
      <td>classification</td>
      <td>None</td>
      <td>None</td>
      <td>Boruta</td>
      <td>k-NN</td>
      <td>/workspaces/fseval/examples/comparing-feature-...</td>
      <td>2022-10-22 14:28:31.230633</td>
    </tr>
    <tr>
      <th>31gd56gf</th>
      <td>My synthetic dataset</td>
      <td>10000</td>
      <td>20</td>
      <td>classification</td>
      <td>None</td>
      <td>None</td>
      <td>Chi-Squared</td>
      <td>k-NN</td>
      <td>/workspaces/fseval/examples/comparing-feature-...</td>
      <td>2022-10-22 14:29:19.633012</td>
    </tr>
    <tr>
      <th>a8washm5</th>
      <td>My synthetic dataset</td>
      <td>10000</td>
      <td>20</td>
      <td>classification</td>
      <td>None</td>
      <td>None</td>
      <td>Decision Tree</td>
      <td>k-NN</td>
      <td>/workspaces/fseval/examples/comparing-feature-...</td>
      <td>2022-10-22 14:29:23.459190</td>
    </tr>
    <tr>
      <th>27i7uwg4</th>
      <td>My synthetic dataset</td>
      <td>10000</td>
      <td>20</td>
      <td>classification</td>
      <td>None</td>
      <td>None</td>
      <td>Infinite Selection</td>
      <td>k-NN</td>
      <td>/workspaces/fseval/examples/comparing-feature-...</td>
      <td>2022-10-22 14:29:27.506974</td>
    </tr>
    <tr>
      <th>3velt3b9</th>
      <td>My synthetic dataset</td>
      <td>10000</td>
      <td>20</td>
      <td>classification</td>
      <td>None</td>
      <td>None</td>
      <td>MultiSURF</td>
      <td>k-NN</td>
      <td>/workspaces/fseval/examples/comparing-feature-...</td>
      <td>2022-10-22 14:29:31.758090</td>
    </tr>
    <tr>
      <th>3fdrxlt6</th>
      <td>My synthetic dataset</td>
      <td>10000</td>
      <td>20</td>
      <td>classification</td>
      <td>None</td>
      <td>None</td>
      <td>Mutual Info</td>
      <td>k-NN</td>
      <td>/workspaces/fseval/examples/comparing-feature-...</td>
      <td>2022-10-22 14:35:04.289361</td>
    </tr>
    <tr>
      <th>14lecx0g</th>
      <td>My synthetic dataset</td>
      <td>10000</td>
      <td>20</td>
      <td>classification</td>
      <td>None</td>
      <td>None</td>
      <td>ReliefF</td>
      <td>k-NN</td>
      <td>/workspaces/fseval/examples/comparing-feature-...</td>
      <td>2022-10-22 14:35:08.614262</td>
    </tr>
    <tr>
      <th>3sggjvu3</th>
      <td>My synthetic dataset</td>
      <td>10000</td>
      <td>20</td>
      <td>classification</td>
      <td>None</td>
      <td>None</td>
      <td>Stability Selection</td>
      <td>k-NN</td>
      <td>/workspaces/fseval/examples/comparing-feature-...</td>
      <td>2022-10-22 14:35:59.121416</td>
    </tr>
    <tr>
      <th>dtt8bvo5</th>
      <td>My synthetic dataset</td>
      <td>10000</td>
      <td>20</td>
      <td>classification</td>
      <td>None</td>
      <td>None</td>
      <td>XGBoost</td>
      <td>k-NN</td>
      <td>/workspaces/fseval/examples/comparing-feature-...</td>
      <td>2022-10-22 14:36:23.385401</td>
    </tr>
  </tbody>
</table>
</div>



Let's also read in the `validation_scores`.


```python
validation_scores: pd.DataFrame = pd.read_sql_table("validation_scores", con=con, index_col="id")
validation_scores
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>index</th>
      <th>n_features_to_select</th>
      <th>fit_time</th>
      <th>score</th>
      <th>bootstrap_state</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3lllxl48</th>
      <td>0</td>
      <td>1</td>
      <td>0.004433</td>
      <td>0.7955</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3lllxl48</th>
      <td>0</td>
      <td>2</td>
      <td>0.004227</td>
      <td>0.7910</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3lllxl48</th>
      <td>0</td>
      <td>3</td>
      <td>0.005183</td>
      <td>0.7950</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3lllxl48</th>
      <td>0</td>
      <td>4</td>
      <td>0.003865</td>
      <td>0.7965</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3lllxl48</th>
      <td>0</td>
      <td>5</td>
      <td>0.002902</td>
      <td>0.7950</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>dtt8bvo5</th>
      <td>0</td>
      <td>16</td>
      <td>0.000670</td>
      <td>0.7805</td>
      <td>1</td>
    </tr>
    <tr>
      <th>dtt8bvo5</th>
      <td>0</td>
      <td>17</td>
      <td>0.000480</td>
      <td>0.7725</td>
      <td>1</td>
    </tr>
    <tr>
      <th>dtt8bvo5</th>
      <td>0</td>
      <td>18</td>
      <td>0.003159</td>
      <td>0.7760</td>
      <td>1</td>
    </tr>
    <tr>
      <th>dtt8bvo5</th>
      <td>0</td>
      <td>19</td>
      <td>0.000848</td>
      <td>0.7650</td>
      <td>1</td>
    </tr>
    <tr>
      <th>dtt8bvo5</th>
      <td>0</td>
      <td>20</td>
      <td>0.000565</td>
      <td>0.7590</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>160 rows × 5 columns</p>
</div>



We can now merge them. Notice that we set as the _index_ the experiment ID, so we can use `pd.DataFrame.join` to do this.


```python
validation_scores_with_experiment_info = experiments.join(
    validation_scores
)
validation_scores_with_experiment_info.head(1)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
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
      <th>index</th>
      <th>n_features_to_select</th>
      <th>fit_time</th>
      <th>score</th>
      <th>bootstrap_state</th>
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
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14lecx0g</th>
      <td>My synthetic dataset</td>
      <td>10000</td>
      <td>20</td>
      <td>classification</td>
      <td>None</td>
      <td>None</td>
      <td>ReliefF</td>
      <td>k-NN</td>
      <td>/workspaces/fseval/examples/comparing-feature-...</td>
      <td>2022-10-22 14:35:08.614262</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Cool! That will be all the information that we need. Let's first create an overview for all the rankers we benchmarked.


```python
validation_scores_with_experiment_info \
        .groupby("ranker") \
        .mean(numeric_only=True) \
        .sort_values("score", ascending=False)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>dataset/n</th>
      <th>dataset/p</th>
      <th>index</th>
      <th>n_features_to_select</th>
      <th>fit_time</th>
      <th>score</th>
      <th>bootstrap_state</th>
    </tr>
    <tr>
      <th>ranker</th>
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
      <th>Infinite Selection</th>
      <td>10000.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>10.5</td>
      <td>0.004600</td>
      <td>0.818925</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>XGBoost</th>
      <td>10000.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>10.5</td>
      <td>0.002998</td>
      <td>0.818575</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Decision Tree</th>
      <td>10000.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>10.5</td>
      <td>0.002810</td>
      <td>0.817675</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Stability Selection</th>
      <td>10000.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>10.5</td>
      <td>0.002406</td>
      <td>0.803325</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Chi-Squared</th>
      <td>10000.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>10.5</td>
      <td>0.002548</td>
      <td>0.795975</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>ANOVA F-value</th>
      <td>10000.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>10.5</td>
      <td>0.003745</td>
      <td>0.789275</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Mutual Info</th>
      <td>10000.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>10.5</td>
      <td>0.002314</td>
      <td>0.786475</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Boruta</th>
      <td>10000.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>10.5</td>
      <td>0.002366</td>
      <td>0.518075</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>MultiSURF</th>
      <td>10000.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ReliefF</th>
      <td>10000.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Already, we notice that MultiSURF and ReliefF are missing. This is because the experiments failed. That can happen in a big benchmark! We will ignore this for now and continue with the other Feature Selectors.

👀 We can already observe, that the _average_ classification accuracy is the highest for Infinite Selection. Although it would be premature to say it is the best, this is an indication that it did will for this dataset.

Let's plot the results _per_ `n_features_to_select`. Note, that `n_features_to_select` means a validation step was run using a feature subset of size `n_features_to_select`.


```python
import plotly.express as px

px.line(
    validation_scores_with_experiment_info,
    x="n_features_to_select",
    y="score",
    color="ranker"
)
```


![feature selectors comparison plot](/img/recipes/feature-selectors-comparison-plot.png)


Indeed, we can see XGBoost, Infinite Selection and Decision Tree are solid contenders for this dataset.

🙌🏻

--- 

This has shown how easy it is to do a large benchmark with `fseval`. Cheers!
