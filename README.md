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

A schema is a plain class -- `@h5t.dataset` for a child dataset, `@h5t.group` for a
group (including the root) -- loaded with `h5t.load`:

```python
import dataclasses
import json
from pathlib import Path
from typing import Annotated, Any

import numpy as np

import h5t


@h5t.dataset(data="data")
@dataclasses.dataclass
class Measurement:
    unit: str
    data: h5t.LazyArray                       # lazy detached payload


@h5t.group()
@dataclasses.dataclass
class Nested:
    label: str


@h5t.group(extras="ignore")
@dataclasses.dataclass
class Result:
    version: int                              # implicit HDF5 attribute
    title: Annotated[str, h5t.Name("name")] # renamed attribute
    config: Annotated[
        dict[str, Any],
        h5t.Attr(converter=json.loads),
    ]
    array_attr: Annotated[np.ndarray, h5t.Attr()]
    values: np.ndarray                        # eager dataset payload
    samples: h5t.LazyArray                    # child dataset, no attributes declared
    measurement: Measurement                  # loaded via @h5t.dataset
    nested: Nested                            # recursively loaded group record
    note: str | None                          # absent becomes None
    revision: int = 1                         # absent uses a validated default


result = h5t.load(Result, Path("result.h5"), root="/")
```

Bind a `Mapping`-annotated field with `attrs=` (see below) to receive the node's full
attribute snapshot: an immutable mapping keyed by on-disk HDF5 name, where declared
attributes hold their Pydantic-processed values and undeclared ones retained under
`extras="ignore"` hold the raw h5py values.

A `LazyArray` payload exposes snapshot metadata and explicit data access:

```python
lazy = result.measurement.data
lazy.path, lazy.shape, lazy.dtype, lazy.ndim

complete = lazy.data         # first access reads and caches an ndarray
assert lazy.read() is complete

with lazy.open() as live:
    first_hundred = live[:100]  # fresh file view, useful for slices
```

`h5t.load()` closes every handle on normal and exceptional exits, and so does
`LazyArray.open()`. h5t only ever reads: it builds a record by calling the record type's
own constructor with the loaded values, and offers no way to write one back.

## Field rules

| Annotation | HDF5 representation | Loading behavior |
| --- | --- | --- |
| `@h5t.group()`-decorated class | child group (or the root) | recursively materialized via `record_type(**fields)` |
| `@h5t.dataset(data=...)`-decorated class | child dataset | attributes loaded, `record_type(**fields)` constructed |
| `Annotated[T, Payload("field")]` | child dataset | attributes loaded, `T(**fields)` constructed |
| `LazyArray` (or a subclass) | child dataset | metadata snapshot, payload read on first access |
| `Annotated[<any dataset field>, Eager()]` | child dataset | complete payload cached during loading |
| `np.ndarray` | child dataset | complete payload loaded as an ndarray |
| `Annotated[T, Attr(...)]` | attribute | converter, then Pydantic validation |
| scalar or `Literal[...]` | attribute | Pydantic validation |

`Name("stored-name")` renames any field kind. `Attr` is valid only for attributes and
never combines with `Payload`. `Eager` is valid only for dataset fields; combined with
`Payload` it is allowed only when the payload field is `LazyArray` (it prefetches the
array), since an `np.ndarray` payload is already eager.
A parameterized alias such as `numpy.typing.NDArray[np.floating]` is accepted wherever
`np.ndarray` is; the dtype parameter is not validated. Unsupported collection-shaped
child annotations raise `SchemaError`; dynamic collections are not yet supported.
A `@h5t.dataset` record may declare only attributes plus its one payload field; a
`@h5t.group` record has no such restriction and may reference other schema types,
including itself. Declarations compile at the decorator call, so a `SchemaError` usually
surfaces right there. A `@h5t.group` record that names a record defined later in its
module, or references itself, instead compiles on first use; `@h5t.dataset` has no such
deferral, and does not need one -- being attributes-only, it can never forward-reference
another schema type.

A record type has no reserved field names: `data`, `path`, `shape` and `attrs` are all
free, because h5t contributes no base class for a field name to shadow.

Every `@h5t.dataset`/`@h5t.group` record accepts `extras="ignore"` (the default) or
`extras="forbid"`. A group policy applies to its immediate child and attribute names. A
dataset policy applies to its attributes. Nested schemas keep their own policy, while a
plain `np.ndarray` field never checks the dataset's attributes.

## Foreign record types (`Payload`)

`@h5t.dataset`/`@h5t.group` are themselves sugar over `Payload`, h5t's lower-level
mechanism for loading a child dataset into a plain record type without decorating it --
useful for a third-party type you cannot add a decorator to:

```python
from dataclasses import dataclass

@dataclass
class Measurement:
    unit: str
    data: np.ndarray                          # eager payload

@h5t.group()
@dataclass
class Result:
    measurement: Annotated[Measurement, h5t.Payload("data")]
```

`Payload("data")` names the field holding the payload: annotate it `np.ndarray` for an
eager array, or `h5t.LazyArray` for h5t's usual detached, lazily-read payload. Every
other field of the record type is loaded as a dataset attribute: scalars, `Literal[...]`,
`Attr(converter=...)`, defaults, and `T | None` all behave as they do on a group record.
`Payload`'s own `extras=` (default `"ignore"`) governs the dataset's attributes, since a
type you did not decorate cannot carry its own policy. h5t calls the record type's real
constructor (`__post_init__` runs; an invariant it raises surfaces as `ValidationError`).

Pass `attrs="field"` to bind a `Mapping`-annotated field to the dataset's attrs
snapshot -- validated values for declared names, raw for undeclared. It lists every
declared name, including ones absent from the file that fell back to `None` or to a
default:

```python
from collections.abc import Mapping
from typing import Any

@dataclass
class Measurement:
    unit: str
    data: np.ndarray
    attrs: Mapping[str, Any]

@h5t.group()
@dataclass
class Result:
    measurement: Annotated[Measurement, h5t.Payload("data", attrs="attrs")]
```

A foreign record has no `path`/`shape`/`dtype` of its own unless its `Payload` field is
a `LazyArray` (whose own `.path`/`.shape`/`.dtype` you read through it). An
`Annotated[T, Payload(...)]` use site resolves `T`'s annotations against its module
globals and class dict only; `@h5t.dataset`/`@h5t.group` additionally capture their own
call site's frame, so a function-local record's annotations resolve against function
locals too. A `Payload` field can also name a `@h5t.group`-decorated type directly.

## Snapshot consistency

A loaded model is a snapshot, with one deliberate exception:

- Group and dataset attributes, dataset shape/dtype/path, eager datasets, and plain
  arrays reflect the file during loading (`h5t.load()`).
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
