---
sidebar_position: 2
---

# ToCSVCallback

```python
class fseval.config.callbacks.ToCSVCallback(
    dir: str=MISSING,
    mode: str="a",
)
```


CSV support for fseval. Uploads general information on the experiment to a `experiments` table and provides a hook for uploading custom tables.

**Attributes**:

| | |
|---|---|
| `dir` : str | The directory to save all CSV files to. For example, in this directory a file `experiments.csv` will be created, containing the config of all experiments that were run. |
| `mode` : str | Whether to overwrite or append. Use "a" for appending and "w" for overwriting. |
| | |
