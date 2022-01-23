---
sidebar_label: cv
title: pipeline.cv
---

## CrossValidator Objects

```python
@dataclass
class CrossValidator()
```

#### train\_test\_split

```python
def train_test_split(X, y) -> Tuple[List, List, List, List]
```

Gets train/test split of current fold.

