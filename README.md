# h5t

[![PyPI](https://img.shields.io/pypi/v/h5t.svg)](https://pypi.org/project/h5t/)
[![Python versions](https://img.shields.io/pypi/pyversions/h5t.svg)](https://pypi.org/project/h5t/)
[![CI](https://github.com/binado/h5t/actions/workflows/ci.yml/badge.svg)](https://github.com/binado/h5t/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/binado/h5t/blob/main/LICENSE)

`h5t` loads HDF5 groups into detached, typed Python records. Group attributes and
ordinary NumPy-array fields are materialized while the file is open. Typed datasets
keep snapshot metadata and can read their payload lazily without retaining an open
file descriptor.

## Installation

```bash
pip install h5t     # or: uv add h5t
```

Requires Python 3.11+. `h5py`, `numpy`, and Pydantic v2 are installed automatically.

## Example

```python
import json
from pathlib import Path
from typing import Annotated, Any

import numpy as np

import h5t


class Measurement(h5t.Dataset, extras="forbid"):
    unit: str


class Nested(h5t.Group):
    label: str


class Result(h5t.Group, extras="ignore"):
    version: int                              # implicit HDF5 attribute
    title: Annotated[str, h5t.Name("name")] # renamed attribute
    config: Annotated[
        dict[str, Any],
        h5t.Attr(converter=json.loads),
    ]
    array_attr: Annotated[np.ndarray, h5t.Attr()]
    values: np.ndarray                        # eager dataset payload
    measurement: Measurement                  # lazy detached dataset
    eager_measurement: Annotated[Measurement, h5t.Eager()]
    nested: Nested                            # recursively loaded group
    note: str | None                          # absent becomes None
    revision: int = 1                         # absent uses a validated default


result = Result.from_file(Path("result.h5"), root="/")
```

`result.attrs` and `result.measurement.attrs` are immutable mappings keyed by their
on-disk HDF5 names. Declared attributes contain their Pydantic-processed values;
undeclared attributes retained under `extras="ignore"` contain the raw h5py values.

Typed datasets expose snapshot metadata and explicit data access:

```python
dataset = result.measurement
dataset.path, dataset.shape, dataset.dtype, dataset.ndim

complete = dataset.data      # first access reads and caches an ndarray
assert dataset.read() is complete

with dataset.open() as live:
    first_hundred = live[:100]  # fresh file view, useful for slices
```

Both `from_file()` and `Dataset.open()` close every handle on normal and exceptional
exits. Schema objects cannot be directly constructed, written, or serialized by h5t.

## Field rules

| Annotation | HDF5 representation | Loading behavior |
| --- | --- | --- |
| `Group` subclass | child group | recursively materialized |
| `Dataset` or subclass | child dataset | metadata/attrs snapshot, payload lazy |
| `Annotated[DatasetSubtype, Eager()]` | child dataset | complete payload cached during loading |
| `np.ndarray` | child dataset | complete payload loaded as an ndarray |
| `Annotated[T, Attr(...)]` | attribute | converter, then Pydantic validation |
| scalar or `Literal[...]` | attribute | Pydantic validation |

`Name("stored-name")` renames any field kind. `Attr` is valid only for attributes and
`Eager` only for typed datasets. A parameterized alias such as
`numpy.typing.NDArray[np.floating]` is accepted wherever `np.ndarray` is; the dtype
parameter is not validated. Unsupported collection-shaped child annotations raise
`SchemaError`; dynamic collections are not yet supported. Declarations are compiled at
the `class` statement, so a `SchemaError` surfaces there. A schema class that names a
class defined later in its module instead compiles on first use.

Each `Group` and `Dataset` subclass accepts `extras="ignore"` (the default) or
`extras="forbid"`. A group policy applies to its immediate child and attribute
names. A typed dataset policy applies to its attributes. Nested schemas keep their own
policy, while a plain `np.ndarray` field never checks the dataset's attributes.

## Snapshot consistency

A loaded model is a snapshot, with one deliberate exception:

- Group and dataset attributes, dataset shape/dtype/path, eager datasets, and plain
  arrays reflect the file during `from_file()`.
- A lazy dataset's first `.data`/`.read()` observes the file at that later moment and
  caches the resulting array permanently.
- `.open()` always opens the current file and current dataset. It may therefore observe
  replacements or fail if the source was changed or deleted.

## CLI

```bash
h5t check result.h5 --schema mypackage.schemas:Result --root /results/latest
```

Exit status is 0 on success, 1 for the first file/schema mismatch, and 2 for import,
declaration, usage, or I/O errors.

## Development

```bash
uv sync
uv run pytest
uv run ty check
uv run ruff check .
```
