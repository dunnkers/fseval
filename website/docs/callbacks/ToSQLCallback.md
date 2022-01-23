---
sidebar_position: 0
---

# ToSQLCallback

```python
class fseval.config.callbacks.ToSQLCallback(
    engine_config: EngineConfig=MISSING,
    if_table_exists: str="append"
)
```

SQL support for fseval. Uploads general information on the experiment to a `experiments` table and provides a hook for uploading custom tables.

**Attributes**:

| | |
|---|---|
| `engine` : [EngineConfig](#engineconfig) | All keyword arguments to pass to SQLAlchemy's [`create_engine`](https://docs.sqlalchemy.org/en/14/core/engines.html#sqlalchemy.create_engine) function. |
| `if_table_exists` : str | What to do when a table of the specified name already exists. Can be 'fail', 'replace' or 'append'. By default is 'append'. For more info, see the [Pandas.DataFrame#to_sql](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_sql.html) function. |
| | |



    
    
## `EngineConfig`

```python
class EngineConfig(
    url: str=MISSING
)
```

A type definition whatever we are passing to SQLAlchemy's [create_engine](https://docs.sqlalchemy.org/en/14/core/engines.html#sqlalchemy.create_engine) function.

**Attributes**:

| | |
|---|---|
| `url` : str | The database URL. Is of type RFC-1738, e.g. `dialect+driver://username:password@host:port/database` See the SQLAlchemy documentation for more information; https://docs.sqlalchemy.org/en/14/core/engines.html#database-urls |
| | |