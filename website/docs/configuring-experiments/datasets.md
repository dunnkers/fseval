---
sidebar_position: 1
---

# DatasetConfig
Datasets can be loaded with well-defined dataset configs. The dataset config looks as follows:

```python
@dataclass
class DatasetConfig:
    """
    Attributes:
        name (str): human-readable name of dataset.
        task (Task): either Task.classification or Task.regression.
        adapter: dataset adapter. must be of fseval.types.AbstractAdapter type,
            i.e. must implement a get_data() -> (X, y) method. Can also be a callable;
            then the callable must return a tuple (X, y).
        adapter_callable: adapter class callable. the function to be called on the
            instantiated class to fetch the data (X, y). is ignored when the target
            itself is a function callable.
        feature_importances (Optional[Dict[str, float]]): weightings indicating relevant
            features or instances. should be a dict with each key and value like the
            following pattern:
                X[<numpy selector>] = <float>
            Example:
                X[:, 0:3] = 1.0
            which sets the 0-3 features as maximally relevant and all others
            minimally relevant.
        group (Optional[str]): an optional group attribute, such to group datasets in
            the analytics stage.
        domain (Optional[str]): dataset domain, e.g. medicine, finance, etc.
    """

    name: str = MISSING
    task: Task = MISSING
    adapter: Any = MISSING
    adapter_callable: str = "get_data"
    feature_importances: Optional[Dict[str, float]] = None
    # optional tags
    group: Optional[str] = None
    domain: Optional[str] = None
    # runtime properties: will be set once dataset is loaded, no need to configure them.
    n: Optional[int] = None
    p: Optional[int] = None
    multioutput: Optional[bool] = None

    # required for instantiation
    _target_: str = "fseval.pipeline.dataset.DatasetLoader"
    _recursive_: bool = False  # prevent adapter from getting initialized
```

## Available adapters
Built-in, the following adapters are available.

### OpenML
To load datasets from [OpenML](https://www.openml.org/), use the `fseval.adapters.openml.OpenML` adapter.
```python
@dataclass
class OpenMLDataset:
    _target_: str = "fseval.adapters.openml.OpenML"
    dataset_id: int = MISSING
    target_column: str = MISSING
    drop_qualitative: bool = False
```

### Weights and Biases
To load a dataset from [Weights and Biases](https://wandb.ai/) [artifacts](https://docs.wandb.ai/guides/artifacts), use the `fseval.adapters.wandb.Wandb` adapter.

```python
@dataclass
class WandbDataset:
    _target_: str = "fseval.adapters.wandb.Wandb"
    artifact_id: str = MISSING
```


### ⚙️ Custom Adapters
To load datasets from different sources, we can use different **adapters**. You can create an adapter by implementing this interface:

```python
class AbstractAdapter(ABC, BaseEstimator):
    @abstractmethod
    def get_data(self) -> Tuple[List, List]:
        ...
```

For example, the _Weights and Biases_ adapter is implemented like so:

```python title="benchmark.py"
import wandb

@dataclass
class WandbAdapter(AbstractAdapter):
    def get_data(self) -> Tuple[List, List]:
        api = wandb.Api()
        artifact = api.artifact(self.artifact_id)
        X = artifact.get("X").data
        Y = artifact.get("Y").data
        return X, Y
```

And then loading an artifact:

```yaml
name: Switch (Chen et al.)
task: regression
adapter:
  _target_: benchmark.WandbAdapter
  artifact_id: dunnkers/synthetic-datasets/switch:v0
feature_importances:
  X[:5000, 0:4]: 1.0
  X[5000:, 4:8]: 1.0
```


## Example dataset configurations
Datasets can be defined using YAML or [Structured Configs](https://hydra.cc/docs/tutorials/structured_config/intro/).

### Example YAML config
We can define the Iris dataset using OpenML:

```yaml title="conf/dataset/iris.yaml"
name: Iris Flowers
task: classification
adapter:
  _target_: fseval.adapters.openml.OpenML
  dataset_id: 61
  target_column: class
```

But we can also use _functions_ as adapters, as long as they return a tuple `(X, y)`. e.g. using `sklearn.datasets.make_classification` as an adapter:

```yaml title="conf/dataset/some_synthetic_dataset.yaml"
name: My synthetic dataset
task: classification
adapter:
  _target_: sklearn.datasets.make_classification
  n_samples: 10000
  n_informative: 2
  n_classes: 2
  n_features: 20
  n_redundant: 0
  random_state: 0
  shuffle: false
feature_importances:
  X[:, 0:2]: 1.0
```

For more examples, see [this](https://github.com/dunnkers/fseval/tree/master/tests/integration/conf/dataset) directory.

### Example Structured config
Any dataset can also be configured using Python code. Like so:

```python
from hydra.core.config_store import ConfigStore
from fseval.config import DatasetConfig
from fseval.types import Task
from fseval.adapters.openml import OpenMLDataset

cs = ConfigStore.instance()

cs.store(
    group="dataset",
    name="iris",
    node=DatasetConfig(
        name="iris",
        task=Task.classification,
        adapter=OpenMLDataset(dataset_id=61, target_column="class"),
    ),
)
```
