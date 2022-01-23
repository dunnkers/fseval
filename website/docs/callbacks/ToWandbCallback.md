---
sidebar_position: 1
---

# ToWandbCallback

```python
class fseval.config.callbacks.ToWandbCallback(
    log_metrics: bool=True,
    wandb_init_kwargs: Dict[str, Any]=field(default_factory=lambda: {}),
)
```

Support for exporting the job config and result tables to Weights and Biases.
    
**Attributes**:

| | |
|---|---|
| `log_metrics` : bool | Whether to log metrics. In the case of a resumation run, a user might probably not want to log metrics, but just update the tables instead. |
| `wandb_init_kwargs` : Dict[str, Any] | Any additional settings to be passed to `wandb.init()`. See the function signature for details; https://docs.wandb.ai/ref/python/init |
| | |