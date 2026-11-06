# AGENTS.md

Guidance for AI agents working in this repository.

## Commands

```bash
uv sync                                  # install deps (uv sync --locked in CI)
uv run pytest                            # full test suite
uv run pytest tests/test_compile.py::test_name   # single test
uv run --python 3.14 pytest              # PEP 649 paths; skipped on the default interpreter
uv run ty check                          # type check (Astral's ty, not mypy/pyright)
uv run ruff check .                      # lint
uv run ruff format --check src tests     # format check, as CI runs it
just docs build                          # Sphinx docs; warnings are errors (-W)
```

`[tool.ty.src]` restricts type checking to `src/h5t` and `tests/acceptance_example.py`. `prek.toml`
runs ruff-check, ruff-format and ty as pre-commit hooks. Commits follow Conventional Commits.

Public docstrings are numpy-style and rendered by Sphinx (`docs/`, built with `-W`), so keep
them user-facing; put internal reasoning in `#` comments.

## Architecture

`h5t` turns `h5t.Group[T]` subclasses (`pydantic.BaseModel` subclasses) into loaders
(`Group.model_validate_hdf5`) and writers (`group.model_dump_hdf5`) for HDF5 groups. Loaded
groups are **detached**: every file handle is closed before `model_validate_hdf5` returns; only a
`Dataset[T]` keeps its `filename`/`path` so slices or the full array can be read lazily on
demand. Both `Group[T]` and `Dataset[T]` are generic over `T` (defaulting to `Mapping[str, Any]`
or accepting a `pydantic.BaseModel` subclass for strongly typed `.attrs: T`).

Dependency direction: `_cli`/`_write` → `_compile` → `_array`/`_spec`/`_errors`. The one
reverse edge is `Group.model_dump_hdf5`, which imports `_write` inside the method to avoid a
real import cycle (the style guide allows deferred imports for that).

- **`_spec.py`** — `Glob` marker and the compiled IR: `ArraySpec`, `AttrFieldSpec`, `AttrsSpec`,
  `FieldSpec`, `GroupSpec`, `MemberKind` (`ARRAY` / `DATASET` / `GROUP` / `COLLECTION`),
  `Extras`, plus error-path formatters.
- **`_array.py`** — `Dataset[T]`, the detached lazily-read dataset with typed `.attrs: T`,
  `h5py`-style indexing (`ds[()]`, `ds[:]`, `ds[:10]`), `.data`, `.read()`, `.open()`, and
  `__array__()`.
- **`_compile.py`** — `Group[T]`, `_GroupMeta` metaclass, annotation → `FieldSpec` / `AttrsSpec`
  compilation, `jaxtyping` Pydantic core-schema hook, and the loader (`model_validate_hdf5`).
- **`_write.py`** — `Group.model_dump_hdf5`: a *plan* phase that validates an in-memory `Group`
  tree without touching the file, then an *apply* phase that writes and rolls back on failure.
- **`_cli.py`** — `h5t check`: import `pkg.mod:Class`, call `Class.model_validate_hdf5`, exit 0
  (ok) / 1 (`ValidationError`) / 2 (import, schema, usage, or I/O error).

### Compilation: eager, with deferral for `Group` forward references

`_GroupMeta` compiles `Group` subclasses at class creation. Annotations resolve against the
module globals, then a `localns` built from the defining caller's locals (used once, never
stored), the explicit `localns=` class keyword argument, and the class's own name (so
self-reference works).

- A bad declaration raises `SchemaError` at class definition.
- A `Group` subclass whose annotation raises `NameError` is stored as a `_PendingGroup` on
  `cls.__h5t_group__` and retried later. Retries see only the explicit `localns`, not the
  defining function's locals — a function-scope group that defers *and* uses a local type must
  pass `class MyGroup(h5t.Group, localns={...}):`.
- A `NameError` raised by a *function the annotation calls* is a bug, not a forward reference
  (`_raised_by_annotation` distinguishes them by the raising frame).
- `_ensure_group_compiled` is the compile-time retry point; `_compiled_group` wraps it for load
  time and turns a still-unresolved name into `SchemaError`.
- On 3.14, use `annotationlib.get_annotations(cls, format=Format.VALUE)`. Do **not** use
  `Format.STRING`/`FORWARDREF`: they run the user's annotation helper inside a swallowing handler,
  hiding its `NameError`.

### Naming

Internal state is prefixed `_h5t_`; the compiled group spec is stored on `cls.__h5t_group__`
(set via `setattr`, deliberately not inherited). Annotations starting with `_` are skipped as
fields. Group fields represent child HDF5 nodes (attributes live on `.attrs: T`); only repeated
*HDF5* names within a group or attributes model are rejected.

### jaxtyping arrays

jaxtyping does the checking; h5t never parses a shape string. `_unwrap_jaxtyping` rewrites an
`AbstractArray` subclass to its backing `np.ndarray`/`Dataset[T]` early in compilation and keeps
the original as an `ArraySpec`. `_check_array` calls the non-public `hint.__instancecheck_str__`
on a metadata-only candidate (a zero-stride `_ghost` array, or a `Dataset`), so no data is read.
A guard test fails if jaxtyping drops that method. Each `_load_group` / `_load_attrs` call opens
its own `jaxtyped("context")` scope; the writer and `Group`'s wrap validator mirror this.

### Array annotations see through type aliases

numpy ≥ 2.5 defines `npt.NDArray` as a PEP 695 `TypeAliasType`, so `typing.get_origin` returns the
alias rather than `np.ndarray`. `_is_ndarray_annotation` expands `__value__` (depth-bounded) until
it reaches a real origin. numpy 2.5 needs Python ≥ 3.12, so a failure here that looks
Python-version-dependent is really numpy-version-dependent.

### Validation

Pydantic v2 `TypeAdapter` is built once per attribute/child field with
`arbitrary_types_allowed=True`. Attribute fields use standard Pydantic `Field(alias=...)`,
`BeforeValidator`, and `PlainSerializer`. Errors are fail-fast and path-aware.
Hierarchy: `H5TError` → `SchemaError`, `ValidationError`.

Missing members go through `_missing_attr_value` / `_missing_child_value`: declared default
(validated; `default_factory` called first), else `None` for `T | None`, else `ValidationError`.

## Testing

`tests/conftest.py` holds the canonical schemas (`Result`, `ResultAttrs`, `Nested`,
`NestedAttrs`, `MeasurementAttrs`), `write_result()`, and `open_fd_count()` (POSIX-only, used by
every fd-leak assertion). Prefer extending these over new ad-hoc fixtures.

- `tests/test_pep649.py` deliberately has **no** `from __future__ import annotations` and skips
  below 3.14; its groups are declared inside test functions. Do not add the future import.
- `tests/acceptance_example.py` is not collected by pytest; its annotated assignments fail
  `ty check` if `model_validate_hdf5` stops returning precisely-typed groups and datasets. Update
  it when the public typing surface changes.
- Handle-lifetime changes in `Group.model_validate_hdf5` or `Dataset.open` must keep the fd-leak
  tests green.
