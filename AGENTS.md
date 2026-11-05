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
```

`[tool.ty.src]` restricts type checking to `src/h5t` and `tests/acceptance_example.py`. `prek.toml`
runs ruff-check, ruff-format and ty as pre-commit hooks. Commits follow Conventional Commits.

## Architecture

`h5t` turns annotated Python classes into loaders (`h5t.load`) and writers (`h5t.save`) for HDF5
groups. Loaded records are **detached**: every handle is closed before `load` returns; only a
`LazyArray` keeps its filename/path so the payload can be read later. A schema is a plain class
(usually a dataclass) decorated with `@h5t.dataset` / `@h5t.group`, or reached through a `Payload`
marker for a type you cannot decorate. h5t adds no base class and constructs every record by
calling the record type's own constructor.

Dependency direction: `_cli`/`_write` → `_compile` → `_array`/`_spec`/`_errors`.

- **`_spec.py`** — public markers (`Name`, `Attr`, `Eager`, `Payload`) and the compiled IR:
  `FieldSpec`, `ClassSpec`, `ForeignSpec` (one compiled record type), plus error-path formatters.
- **`_array.py`** — `LazyArray`, the detached lazily-read payload. Usable as a record's `Payload`
  field or directly as a member annotation.
- **`_compile.py`** — annotation → `FieldSpec` compilation *and* the loader. `MemberKind`
  (ATTRIBUTE / ARRAY / DATASET / GROUP / COLLECTION) selects the load path in `_load_group_values`.
- **`_write.py`** — `h5t.save`: a *plan* phase that mirrors the load functions over an in-memory
  record without touching the file, then an *apply* phase that writes and rolls back on failure.
- **`_cli.py`** — `h5t check`: import `pkg.mod:Class`, call `h5t.load`, exit 0 (ok) / 1
  (`ValidationError`) / 2 (import, schema, usage, or I/O error).

`RecordKind` (DATASET vs GROUP) is load-bearing: a `DATASET` record names one payload field and may
declare only attributes besides; a `GROUP` record has no payload and admits nested groups, datasets
and arrays. "Foreign" in internal names (`_foreign_spec`, `ForeignSpec`) is historical — these are
simply the record types.

### Compilation: eager, with deferral for `GROUP` forward references

The decorators compile at the decorator call. Annotations resolve against the module globals, then
a `localns` built from the decorator caller's locals (used once, never stored), the explicit
`localns=` kwarg, and the class's own name (so self-reference works).

- A bad declaration raises `SchemaError` from the decorator call itself.
- `DATASET` records never defer (attributes only, so no forward references are possible).
- A `GROUP` record whose annotation raises `NameError` is stored as a `_PendingRecord` on
  `cls.__h5t_record__` and retried later. Retries see only the explicit `localns`, not the
  decorating function's locals — a function-scope group that defers *and* uses a local type must
  pass `localns={...}`.
- A `NameError` raised by a *function the annotation calls* is a bug, not a forward reference
  (`_raised_by_annotation` distinguishes them by the raising frame).
- `_ensure_record_compiled` is the compile-time retry point; `_compiled_record` wraps it for load
  time and turns a still-unresolved name into `SchemaError`.
- A `GROUP`-kind member's `ForeignSpec` is resolved at load time, not compile time, because eager
  resolution would recurse forever on `child: Node | None`.
- On 3.14, use `annotationlib.get_annotations(cls, format=Format.VALUE)`. Do **not** use
  `Format.STRING`/`FORWARDREF`: they run the user's annotation helper inside a swallowing handler,
  hiding its `NameError`.

### Naming

Internal state is prefixed `_h5t_`; the only name written onto a user's class is `__h5t_record__`
(set via `setattr`, deliberately not inherited). Annotations starting with `_` are skipped as
fields. Record fields may use any name (`data`, `attrs`, `path`, ...); only repeated *HDF5* names
within a record are rejected.

### jaxtyping arrays

jaxtyping does the checking; h5t never parses a shape string. `_unwrap_jaxtyping` rewrites an
`AbstractArray` subclass to its backing `np.ndarray`/`LazyArray` early in compilation and keeps the
original as an `ArraySpec`. `_check_array` calls the non-public `hint.__instancecheck_str__` on a
metadata-only candidate (a zero-stride `_ghost` array, or a `LazyArray`), so no data is read. A
guard test fails if jaxtyping drops that method. Each `_load_group_values` / `_load_foreign_dataset`
call opens its own `jaxtyped("context")` scope; the writer mirrors this.

### Array annotations see through type aliases

numpy ≥ 2.5 defines `npt.NDArray` as a PEP 695 `TypeAliasType`, so `typing.get_origin` returns the
alias rather than `np.ndarray`. `_is_ndarray_annotation` expands `__value__` (depth-bounded) until
it reaches a real origin. numpy 2.5 needs Python ≥ 3.12, so a failure here that looks
Python-version-dependent is really numpy-version-dependent.

### Validation

Pydantic is used only through `TypeAdapter` (never `BaseModel`), built once per field with
`arbitrary_types_allowed=True`. Attribute order: raw h5py value → `Attr(converter=)` (failure ⇒
`ConversionError`) → adapter (failure ⇒ `ValidationError`). Errors are fail-fast and path-aware.
Hierarchy: `H5TError` → `SchemaError`, `ValidationError` → `ConversionError`.

Missing members go through `_missing_value`: declared default (validated; `default_factory` called
first), else `None` for `T | None`, else `ValidationError`.

Pydantic/dataclass quirks that shaped the design:

1. `TypeAdapter(Dataclass, config=...)` raises, and an unconfigured one would revalidate fields, so
   `_build_record_adapter` uses an `InstanceOf` adapter instead.
2. `dataclasses.field(default_factory=...)` leaves no class attribute, so defaults are read from
   `dataclasses.fields()` for dataclasses; for other types, from the declaring class, then the
   constructor signature.
3. Records are built with the real constructor via `_construct_record` (never `object.__new__`).
   Only an argument-binding `TypeError` is a `SchemaError`; anything raised after user code starts
   (including `__post_init__`) becomes a `ValidationError`.

A `MemberKind.DATASET` field with `foreign is None` is a bare `LazyArray` member constructed
directly; only `Eager()` is meaningful on it.

## Testing

`tests/conftest.py` holds the canonical schemas (`Result`, `Nested`, `EagerMeasurement`,
`LazyMeasurement`), `write_result()`, and `open_fd_count()` (POSIX-only, used by every fd-leak
assertion). Prefer extending these over new ad-hoc fixtures.

- `tests/test_pep649.py` deliberately has **no** `from __future__ import annotations` and skips
  below 3.14; its records are declared inside test functions. Do not add the future import.
- `tests/acceptance_example.py` is not collected by pytest; its annotated assignments fail
  `ty check` if `h5t.load` stops returning precisely-typed records. Update it when the public
  typing surface changes.
- Handle-lifetime changes in `h5t.load` or `LazyArray.open` must keep the fd-leak tests green.
