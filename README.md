# h5t

[![PyPI](https://img.shields.io/pypi/v/h5t.svg)](https://pypi.org/project/h5t/)
[![Python versions](https://img.shields.io/pypi/pyversions/h5t.svg)](https://pypi.org/project/h5t/)
[![CI](https://github.com/binado/h5t/actions/workflows/ci.yml/badge.svg)](https://github.com/binado/h5t/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/h5t/badge/?version=latest)](https://h5t.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/binado/h5t/blob/main/LICENSE)

`h5t` loads HDF5 groups into detached, typed Python records, and writes them back with
the same schema. Group attributes and ordinary NumPy-array fields are materialized while
the file is open. Typed datasets keep snapshot metadata and can read their payload lazily
without retaining an open file descriptor.

## Installation

```bash
pip install h5t     # or: uv add h5t
```

Requires Python 3.11+. `h5py`, `numpy`, `jaxtyping` and Pydantic v2 are installed automatically.

## Usage

A schema is a plain class: `@h5t.dataset` for a child dataset, `@h5t.group` for a group
(including the root).

```python
import dataclasses
from typing import Annotated

import numpy as np

import h5t


@h5t.dataset(data="data")
@dataclasses.dataclass
class Measurement:
    unit: str
    data: h5t.LazyArray                       # lazy, detached payload


@h5t.group()
@dataclasses.dataclass
class Result:
    version: int                              # HDF5 attribute
    title: Annotated[str, h5t.Name("name")]   # attribute stored as "name"
    values: np.ndarray                        # child dataset, read eagerly
    measurement: Measurement                  # child dataset with attributes


lazy = h5t.LazyArray("", "", (4,), np.dtype("f8"), data=np.linspace(0, 1, 4))
record = Result(1, "demo", np.arange(3.0), Measurement("m", lazy))

h5t.save(record, "result.h5", mode="w")
result = h5t.load(Result, "result.h5")
print(result.measurement.data.data)           # the payload is read on first access
```

Full documentation: <https://h5t.readthedocs.io/en/latest/>. The `h5t check` command
validates a file against a schema from the shell; see the
[CLI guide](https://h5t.readthedocs.io/en/latest/user_guide/cli.html).

## Development

```bash
uv sync
uv run pytest
just docs build
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.
