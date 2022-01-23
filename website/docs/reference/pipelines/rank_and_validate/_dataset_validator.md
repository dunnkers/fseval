---
sidebar_label: _dataset_validator
title: pipelines.rank_and_validate._dataset_validator
---

## DatasetValidator Objects

```python
@dataclass
class DatasetValidator(Experiment,  RankAndValidatePipeline)
```

Validates an entire dataset, given a fitted ranker and its feature ranking. Fits
at most `p` feature subsets, at each step incrementally including more top-features.

