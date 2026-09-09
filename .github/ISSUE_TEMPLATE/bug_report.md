---
name: Bug report
about: Report unexpected behavior in h5t
title: ""
labels: bug
---

**Describe the bug**

A clear description of what went wrong, including the full error message/traceback if any.

**Minimal schema and reproduction**

```python
import h5t

class Result(h5t.Group):
    ...  # minimal schema that reproduces the issue
```

Steps or code used to produce the HDF5 file (or attach a minimal `.h5` file), and the call that
triggers the bug (e.g. `Result.from_file(...)`).

**Expected behavior**

What you expected to happen instead.

**Versions**

- `h5t`:
- Python:
- `h5py`:
- `numpy`:
- `pydantic`:
- OS:
