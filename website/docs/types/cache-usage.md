---
sidebar_position: 0
---

# CacheUsage

```python

class CacheUsage(Enum):
    """
    Determines how cache usage is handled. In the case of **loading** caches:

    - `allow`: program might use cache; if found and could be restored
    - `must`: program should fail if no cache found
    - `never`: program should not load cache even if found

    When **saving** caches:
    - `allow`: program might save cache; no fatal error thrown when fails
    - `must`: program must save cache; throws error if fails (e.g. due to out of memory)
    - `never`: program does not try to save a cached version
    """

    allow = 1
    must = 2
    never = 3
```