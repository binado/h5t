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
| `Float[np.ndarray, "n n"]`, `Float[LazyArray, "n _"]` (jaxtyping) | child dataset | dtype and shape checked against the file's metadata before any read; otherwise as `np.ndarray` / `LazyArray` |
| `dict[str, T]` / `Mapping[str, T]` | child group named by the field | one item per child of that group, as a plain `dict`; `T` is any group/dataset record, `Payload` type, `LazyArray`, `np.ndarray` or jaxtyping array; filter with `Glob` |
| `Annotated[T, Attr(...)]` | attribute | converter, then Pydantic validation |
| scalar or `Literal[...]` | attribute | Pydantic validation |

`Name("stored-name")` renames any field kind. `Attr` is valid only for attributes and
never combines with `Payload`. `Eager` is valid only for dataset fields; combined with
`Payload` it is allowed only when the payload field is `LazyArray` (it prefetches the
array), since an `np.ndarray` payload is already eager.
A parameterized alias such as `numpy.typing.NDArray[np.floating]` is accepted wherever
`np.ndarray` is; for `NDArray` the dtype parameter is not validated (use jaxtyping, below,
to check it). Unsupported collection-shaped
child annotations raise `SchemaError`.

### Dynamic collections (`dict[str, T]`, `Glob`)

Real files often hold one group or dataset per sample, with names a schema cannot list.
A `dict[str, T]` or `Mapping[str, T]` field (no `Attr`) mirrors the HDF5 tree: it names
**one child group** (`Name` is honoured), and its value is a plain `dict` with one item per
child of that group, in HDF5 iteration order. `Annotated[..., h5t.Glob("chain_?")]` keeps
only children whose name matches the shell-style pattern (case-sensitive; it may not be empty
or contain `/`). Attributes on the collection's group are ignored.

```python
@h5t.group()
@dataclass
class Run:
    snapshots: dict[str, Snapshot]
    chains: Annotated[dict[str, h5t.LazyArray], h5t.Glob("chain_?")]
    raw: Annotated[dict[str, Snapshot], h5t.Name("Raw Data")] = field(default_factory=dict)
```

A missing group follows the usual rules (`| None`, a default or `default_factory`, else a
`ValidationError`). Keys must be `str`; items cannot be scalars, optional, nested
collections, or carry `Name`/`Attr`/`Glob`. Bare array items share the owner record's
jaxtyping dimension bindings; record items bind independently. Layouts with per-sample
groups at the file *root* are not covered yet.
A [jaxtyping](https://docs.kidger.site/jaxtyping/) annotation over `np.ndarray` or
`LazyArray` has its dtype and shape checked by jaxtyping itself, from the dataset's HDF5
metadata and without reading the data or casting it. `Float` accepts any float width and
`Float64` only float64 (a file's byte order is ignored); `Shaped` accepts any dtype, and a
custom `AbstractDtype` subclass works too. The whole shape grammar is jaxtyping's (`...`,
`*batch`, `#n`, `n+1`, ...). A named dimension binds on first use and must agree across
every array field of the same record (a nested record starts fresh, and a `@h5t.dataset`
record is its own scope), so `fisher: Float[np.ndarray, "n n"]` with
`fiducials: Float[np.ndarray, "n"]` rejects a mismatched file as
`/fiducials: the size of dimension n is 4 which does not equal the existing value of 3`.
A `@h5t.dataset` record may declare only attributes plus its one payload field; a
`@h5t.group` record has no such restriction and may reference other schema types,
including itself. Declarations compile at the decorator call, so a `SchemaError` usually
surfaces right there. A `@h5t.group` record that names a record defined later in its
module, or references itself, instead compiles on first use; `@h5t.dataset` has no such
deferral, and does not need one -- being attributes-only, it can never forward-reference
another schema type.

A record declared inside a function sees that function's locals when the decorator runs,
so ordinary function-local records and self-references just work. A deferred
`@h5t.group` record is retried later without them: if it both forward-references a name
and uses a function-local type, pass the type explicitly with
`@h5t.group(localns={"Child": Child})` (`@h5t.dataset` accepts `localns=` too).

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
