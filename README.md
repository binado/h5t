# h5t

[![PyPI](https://img.shields.io/pypi/v/h5t.svg)](https://pypi.org/project/h5t/)
[![Python versions](https://img.shields.io/pypi/pyversions/h5t.svg)](https://pypi.org/project/h5t/)
[![CI](https://github.com/binado/h5t/actions/workflows/ci.yml/badge.svg)](https://github.com/binado/h5t/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/h5t/badge/?version=latest)](https://h5t.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/binado/h5t/blob/main/LICENSE)

`h5t` validates and loads HDF5 groups into detached, typed Pydantic models, and writes them
back using the same schema. Group and dataset attributes are modeled with `pydantic.BaseModel`
via `h5t.Group[T]` and `h5t.Dataset[T]`, while child datasets remain lazy and read slices on
demand without retaining open file descriptors.

## Installation

```bash
pip install h5t     # or: uv add h5t
```

Requires Python 3.11+. `h5py`, `numpy`, `jaxtyping` and Pydantic v2 are installed automatically.

## Usage

Subclass `h5t.Group` (or `h5t.Group[Attrs]` where `Attrs` is a `pydantic.BaseModel` subclass)
to describe an HDF5 group. Use `h5t.Dataset[Attrs]` (or bare `h5t.Dataset`) for lazy datasets
with typed attributes, or `np.ndarray` for eager array datasets:

```python
from typing import Annotated

import numpy as np
from jaxtyping import Float
from pydantic import BaseModel, Field

import h5t


class MeasurementAttrs(BaseModel):
    unit: str


class ResultAttrs(BaseModel):
    version: int
    title: Annotated[str, Field(alias="name")]


class Result(h5t.Group[ResultAttrs]):
    values: Float[np.ndarray, "n"]                       # child dataset, read eagerly
    measurement: Float[h5t.Dataset[MeasurementAttrs], "n"]  # lazy child dataset + typed attrs


record = Result(
    attrs=ResultAttrs(version=1, title="demo"),
    values=np.arange(4.0),
    measurement=h5t.Dataset(np.linspace(0, 1, 4), MeasurementAttrs(unit="m")),
)

record.model_dump_hdf5("result.h5", mode="w")
result = Result.model_validate_hdf5("result.h5")
print(result.attrs.title, result.measurement.attrs.unit)
print(result.measurement[:2])                            # slices are read lazily on demand
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
