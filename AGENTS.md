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

Dependency direction: `_cli`/`_write` → `_compile` → `_array`/`_errors` → `_h5view`
(a leaf: imports only h5py/numpy). The one
reverse edge is `Group.model_dump_hdf5`, which imports `_write` inside the method to avoid a
real import cycle (the style guide allows deferred imports for that).

- **`_errors.py`** — `H5TError`, `SchemaError`, `ValidationError`, and HDF5 path formatters.
- **`_h5view.py`** — `_GroupView`, a read-only `Mapping` over an open `h5py.Group`, plus the raw
  attribute helpers (`_normalized_attrs`) and `_ghost`.
- **`_array.py`** — `Dataset[T]`, the detached lazily-read dataset with typed `.attrs: T`,
  `h5py`-style indexing (`ds[()]`, `ds[:]`, `ds[:10]`), `.data`, `.open()`, and
  `__array__()`.
- **`_compile.py`** — `Group[T]`, stateless declaration checks over Pydantic metadata,
  annotation helpers, jaxtyping core-schema hook, and the loader (`model_validate_hdf5`).
- **`_write.py`** — `Group.model_dump_hdf5`: a *plan* phase that walks an in-memory `Group`
  tree without touching the file (it trusts construction-time validation and checks only what
  HDF5 imposes), then an *apply* phase that writes it.
- **`_cli.py`** — `h5t check`: import `pkg.mod:Class`, call `Class.model_validate_hdf5`, exit 0
  (ok) / 1 (`ValidationError`) / 2 (import, schema, usage, or I/O error).

### Model lifecycle: owned by Pydantic

`Group` is an ordinary Pydantic model with no custom metaclass, compiled schema IR,
caches, or pending-model state. HDF5 operations check `model_fields` directly, rebuilding
incomplete models with `model_rebuild()`. `_check_group` walks reachable declarations
with an operation-local visited set before opening a file; recursive schemas terminate.

- HDF5 declaration restrictions raise `SchemaError` at load/write time.
- Pydantic model-building and unresolved-annotation errors propagate unchanged.
- Neither `np.ndarray` nor `jaxtyping.AbstractArray` is monkey-patched. Instead,
  `_prepare_group_fields` (called in `Group.__get_pydantic_core_schema__`) and
  `_prepare_attrs_model` recursively wrap array leaves in `model_fields` with `Annotated`
  markers (`_NdarraySchema` and `_JaxtypingSchema`). Foreign models h5t never sees (jax,
  torch, ...) are never touched; the unsupported-type `SchemaError` comes from
  `_unwrap_jaxtyping` at load/write time.
- `localns=` is removed. Resolve local dependencies explicitly with
  `Model.model_rebuild(_types_namespace={...})`; do not inspect frames or clear namespaces.
- The writer checks runtime group and attribute model declarations during planning,
  including subclass fields in direct children and collection items.

### Naming

Internal state is prefixed `_h5t_`. Pydantic excludes annotations starting with `_` from
fields. Group fields represent child HDF5 nodes (attributes live on `.attrs: T`); only repeated
*HDF5* names within a group or attributes model are rejected.

### jaxtyping arrays

jaxtyping does the checking; h5t never parses a shape string. `_unwrap_jaxtyping` rewrites an
`AbstractArray` subclass to its backing `np.ndarray`/`Dataset[T]` during annotation inspection. The
Pydantic core-schema hook (`_jaxtyping_pydantic_schema`) calls the non-public
`hint.__instancecheck_str__` on a metadata-only candidate (a zero-stride `_ghost` array, or a
`Dataset`), so no data is read. A guard test fails if jaxtyping drops that method. Dimension
bindings are scoped per group: `Group`'s wrap validator opens one `jaxtyped("context")` for the
children, and the `attrs` field has its own scope (`Group._h5t_attrs_scope`) so attribute dims
never bind with child dims. A jaxtyping `np.ndarray` field checks a `_ghost` of an `h5py.Dataset`
*before* reading it; a `Dataset`-backed one builds the lazy `Dataset` and checks that.

### Array annotations see through type aliases

numpy ≥ 2.5 defines `npt.NDArray` as a PEP 695 `TypeAliasType`, so `typing.get_origin` returns the
alias rather than `np.ndarray`. `_is_ndarray_annotation` expands `__value__` (depth-bounded) until
it reaches a real origin. numpy 2.5 needs Python ≥ 3.12, so a failure here that looks
Python-version-dependent is really numpy-version-dependent.

### Validation: one pass, type-owned readers

`Group.model_validate_hdf5` opens the file and calls `cls.model_validate(_GroupView(node))`
once. The view hands Pydantic raw material: child groups as nested views, child datasets as the
raw `h5py.Dataset`, and the attributes under `ATTRS_KEY` (`"/attrs"`, not a legal child name;
`Group.attrs` has `validation_alias=ATTRS_KEY`, so in-memory `Group(attrs=...)` still works).
Each *type* converts raw input itself:

- `Dataset[T]` builds a lazy `Dataset` from an `h5py.Dataset` (typed attrs are validated in their
  own jaxtyping scope via `handler.generate_schema(attrs_type)`; failures flow out in `loc` and
  are formatted by `_h5_error_path` as `<path>@<attr>`).
- jaxtyping arrays and plain `np.ndarray` fields (including collections and chained PEP 695
  aliases of them): `_rewrite_annotation` wraps each array leaf in `Annotated[..., _JaxtypingSchema | _NdarraySchema]`
  before Pydantic builds the model's core schema, so `_read_h5py` or `_read_array` runs via
  Pydantic's standard `Annotated` hook without walking or rewriting raw `CoreSchema` dicts.
- `Group._h5t_jaxtyping_scope` rejects any attribute on a `extra="forbid"` group with untyped
  attrs, and freezes loaded untyped attrs into a `MappingProxyType`.

Pydantic's semantics apply: `T | None` without a default is required, defaults are not
re-validated unless `validate_default`, and under `extra="ignore"` only declared names are read.
Pydantic's `ValidationError` is a `ValueError`, so anything raised inside a validator must be an
`h5t.ValidationError` (not a `ValueError`) to propagate unchanged.

### Writing: trust construction

`Group.model_dump_hdf5` does not re-validate. Field types, jaxtyping shapes and attribute models
were checked when the instance was built, so `_write._plan_group` walks runtime `model_fields`.
Typed attributes come from `model_dump(mode="python", by_alias=True, include=<declared fields>)`,
so serializers and aliases are Pydantic's; Nullable field annotations determine which attributes holding `None`
are left out (a required `Any` attribute holding `None` is an error, not silently skipped). The
writer still checks what Pydantic cannot know: collection keys, untyped attributes on an
`extra="forbid"` group, and values h5py cannot store. An instance
built with `model_construct` or mutated after construction is written as it is.

Errors keep pydantic's wording; `_h5_error_path` walks the first error's `loc` through the
model fields and annotations to produce the HDF5 path (`/values: Field required`, `/@count: ...`,
`/runs/r1/values: ...`). A `loc` part it cannot place goes into the message. Errors raised
directly as `h5t.ValidationError` already carry their path. `_check_group` checks every
reachable declaration first; unresolved names retain native Pydantic errors.
Hierarchy: `H5TError` → `SchemaError`, `ValidationError`.

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
