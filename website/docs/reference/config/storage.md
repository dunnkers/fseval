---
sidebar_label: storage
title: config.storage
---

## StorageConfig Objects

```python
@dataclass
class StorageConfig()
```

Allows you to define a storage for loading and saving cached estimators, among other
files, like the hydra and fseval configuration in YAML.

**Attributes**:

- `load_dir` _Optional[str]_ - Defines a path to load files from. Must point to
  exactly the directory containing the files, i.e. you should not point to a
  higher-level directory than where the files are. Path can be relative or
  absolute, but an absolute path is recommended.
- `save_dir` _Optional[str]_ - The directory to save files to. Can be relative or
  absolute.

