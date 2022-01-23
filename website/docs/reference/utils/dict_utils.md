---
sidebar_label: dict_utils
title: utils.dict_utils
---

#### dict\_merge

```python
def dict_merge(dct: dict, merge_dct: dict)
```

Recursive dict merge. Inspired by :meth:``dict.update()``, instead of

updating only top-level keys, dict_merge recurses down into dicts nested
to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
``dct``.

Args:
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None


