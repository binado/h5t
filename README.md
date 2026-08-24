# h5t

*A schema layer for HDF5: declare your file format as a Python class, validate files
without reading dataset payloads, and share the class so downstream users load files
correctly.*

## Why

HDF5 in scientific Python is an undocumented nested dict. The information needed to
catch truncated datasets, wrong dtypes, and missing attrs is already in the file's
metadata — names, dtypes, shapes, attrs. `h5t` lets you *state* what you expect and
have it checked, in milliseconds, without reading a single array.

## Example

```python
from typing import Annotated, Literal

import h5t


class Posterior(h5t.Group, dims={"n_samples": h5t.FromAttr("n_samples")}):
    """Posterior samples from one PE run."""

    mass_1: h5t.Dataset[h5t.f8, "n_samples"]
    mass_2: h5t.Dataset[h5t.f8, "n_samples"]
    log_likelihood: h5t.Dataset[h5t.f8, "n_samples"]
    spins: h5t.Dataset[h5t.f8, "n_samples 3"] | None  # optional member
    psd: h5t.Dataset[h5t.f8, "n_freq 2"]

    approximant: str          # attr, by elimination
    f_ref: float = 20.0       # default applied on write


class PEResult(h5t.File, extras="ignore"):
    """LVK-style parameter estimation result."""

    runs: Annotated[
        h5t.Group[Posterior],
        h5t.Keys(pattern=r"C\d+:.*"),
    ]
    format_version: Annotated[
        Literal["1.0"],
        h5t.Name("version"),
    ]
```

```python
with PEResult.open("GW150914.h5") as f:          # validated on open
    m1 = f.runs["C01:IMRPhenomXPHM"].mass_1[:]   # reads only here
```

## CLI

```bash
h5t check GW150914.h5 --schema gwlib.schemas:PEResult
```

## Principles

1. Validation reads metadata, never dataset payloads.
2. The class declares a schema; its instances are lazy views.
3. Datasets are lazy, attrs are eager.
4. h5py's shape, not a new one — mapping semantics plus typed attribute access.
5. Class syntax is a front-end; everything compiles to a plain spec tree.

See [PLAN.md](PLAN.md) for the full design document.

## Development

```bash
uv sync
uv run pytest
uv run mypy
uv run ruff check
```
