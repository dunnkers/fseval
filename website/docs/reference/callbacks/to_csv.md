---
sidebar_label: to_csv
title: callbacks.to_csv
---

## CSVCallback Objects

```python
class CSVCallback(BaseExportCallback)
```

CSV support for fseval. Uploads general information on the experiment to
a `experiments` table and provides a hook for uploading custom tables. Use the
`on_table` hook in your pipeline to upload a DataFrame to a certain database table.

