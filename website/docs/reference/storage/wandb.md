---
sidebar_label: wandb
title: storage.wandb
---

## WandbStorage Objects

```python
@dataclass
class WandbStorage(LocalStorage)
```

Storage for Weights and Biases (wandb), allowing users to save- and
restore files to the service.

**Arguments**:

- `load_dir` - Optional[str] - when set, an attempt is made to load from the
  designated local directory first, before downloading the data off of wandb. Can
  be used to perform faster loads or prevent being rate-limited on wandb.
  
- `save_dir` - Optional[str] - when set, uses this directory to save files, instead
  of the usual wandb run directory, under the `files` subdirectory.
  
- `entity` - Optional[str] - allows you to recover from a specific entity,
  instead of using the entity that is set for the &#x27;current&#x27; run.
  
- `project` - Optional[str] - recover from a specific project.
  
- `run_id` - Optional[str] - recover from a specific run id.
  
- `save_policy` - str - policy for `wandb.save`. Can be &#x27;live&#x27;, &#x27;now&#x27; or &#x27;end&#x27;.
  Determines at which point of the run the file is uploaded.

#### restore

```python
def restore(filename: str, reader: Callable, mode: str = "r") -> Any
```

Given a filename, restores the file either from local disk or from wandb,
depending on the availability of the file. First, the local disk is searched
for the file, taking in regard the `local_dir` value in the
`WandbStorage` constructor. If this file is not found, the file will
be downloaded fresh from wandb servers.

