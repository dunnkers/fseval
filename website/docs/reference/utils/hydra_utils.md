---
sidebar_label: hydra_utils
title: utils.hydra_utils
---

#### get\_config

```python
def get_config(config_module: str, config_name: str, overrides: List[str] = []) -> PipelineConfig
```

Gets the fseval configuration as composed by Hydra. Local .yaml configuration
and defaults are automatically merged.

#### get\_group\_options

```python
def get_group_options(config_module: str, group_name: str, results_filter: Optional[ObjectType] = ObjectType.CONFIG) -> List[str]
```

Gets the options for a certain grouop.

e.g. `get_group_options(&lt;dataset_name&gt;)` returns a list with all
available dataset names for use in fseval.

