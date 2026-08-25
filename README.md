# h5t

*A schema layer for HDF5: declare your file format as a Python class, validate its
structure and domain invariants, and share the class so downstream users load files
correctly.*

## Why

HDF5 in scientific Python is an undocumented nested dict. `h5t` checks the repetitive
parts of a format — names, node kinds, dtypes, and attrs — and lets ordinary Python
express relationships between nodes. Built-in checks read metadata only; custom
validators may also inspect dataset values when the format requires it.

## Example

```python
from typing import Annotated, Literal

import h5t


class Posterior(h5t.Group):
    """Posterior samples from one PE run."""

    mass_1: h5t.Dataset[h5t.f8]
    mass_2: h5t.Dataset[h5t.f8]
    log_likelihood: h5t.Dataset[h5t.f8]
    spins: h5t.Dataset[h5t.f8] | None  # optional member
    psd: h5t.Dataset[h5t.f8]

    n_samples: int
    approximant: str  # attr, by elimination
    f_ref: float = 20.0  # default applied on write

    def validate(self) -> None:
        expected = (self.n_samples,)
        if self.mass_1.shape != expected:
            raise h5t.Invalid(f"mass_1 must have shape {expected}")
        if self.mass_2.shape != expected:
            raise h5t.Invalid(f"mass_2 must have shape {expected}")
        if self.log_likelihood.shape != expected:
            raise h5t.Invalid(f"log_likelihood must have shape {expected}")
        if self.spins is not None and self.spins.shape != (self.n_samples, 3):
            raise h5t.Invalid(f"spins must have shape ({self.n_samples}, 3)")
        if self.psd.ndim != 2 or self.psd.shape[1] != 2:
            raise h5t.Invalid("psd must have shape (n_freq, 2)")


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
with PEResult.open("GW150914.h5") as f:  # validated on open
    m1 = f.runs["C01:IMRPhenomXPHM"].mass_1[:]  # reads only here
```

## CLI

```bash
h5t check GW150914.h5 --schema gwlib.schemas:PEResult
```

## Principles

1. Built-in validation reads metadata; custom validators control whether payloads are read.
2. The class declares a schema; its instances are lazy views.
3. Datasets are lazy, attrs are eager.
4. h5py's shape, not a new one — mapping semantics plus typed attribute access.
5. Relationships between nodes are ordinary Python in `validate(self)` hooks.

## Development

```bash
uv sync
uv run pytest
uv run mypy
uv run ruff check
```
