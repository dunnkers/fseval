---
sidebar_label: to_sql
title: callbacks.to_sql
---

## SQLCallback Objects

```python
class SQLCallback(BaseExportCallback)
```

SQL support for fseval. Uploads general information on the experiment to
a `experiments` table and provides a hook for uploading custom tables. Use the
`on_table` hook in your pipeline to upload a DataFrame to a certain database table.

Support for SQL exports is achieved through using Pandas `df.to_sql` function. This
function, in its turn, then uses SQLAlchemy to export to SQL. Therefore, to use this
callback, it is required you configure the `engine.url` parameter, used to connect
with the database.

