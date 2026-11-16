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

- **`_errors.py`** — `H5TError`, `SchemaError`, `ValidationError`, the HDF5 path formatters, and
  `_located_error`, which turns the first error of a Pydantic `ValidationError` into an
  `h5t.ValidationError` under a known HDF5 path. (It imports only `ATTRS_KEY` from `_h5view`.)
- **`_h5view.py`** — `_GroupView`, a read-only `Mapping` over an open `h5py.Group`, plus the raw
  attribute helpers (`_normalized_attrs`) and `_ghost`.
- **`_array.py`** — `Dataset[T]`, the detached lazily-read dataset with typed `.attrs: T`,
  `h5py`-style indexing (`ds[()]`, `ds[:]`, `ds[:10]`), `.data`, `.open()`, and
  `__array__()`.
- **`_compile.py`** — `Group[T]`, the declaration checks (`_check_declarations` → `_mark`,
  `_check_attrs`), the two array markers, and the loader (`model_validate_hdf5`).
- **`_write.py`** — `Group.model_dump_hdf5`: one pass over an in-memory `Group` tree that
  creates each node as it goes. It trusts construction-time validation and checks only what
  HDF5 imposes. Not transactional: a failure leaves what was already written.
- **`_cli.py`** — `h5t check`: import `pkg.mod:Class`, call `Class.model_validate_hdf5`, exit 0
  (ok) / 1 (`ValidationError`) / 2 (import, schema, usage, or I/O error).

### Model lifecycle: owned by Pydantic

`Group` is an ordinary Pydantic model with no custom metaclass, compiled schema IR,
caches, or pending-model state.

- Declarations are checked **when Pydantic builds the class's schema**:
  `Group.__get_pydantic_core_schema__` calls `_check_declarations(cls)`, which checks only
  that class's own fields. Nested groups run the same hook when Pydantic reaches them, so
  there is no recursion through the model tree and no visited set. Nothing is re-checked at
  load or write time.
- So an HDF5 declaration restriction raises `SchemaError` where the class is defined (or at
  the `model_rebuild()`/first use that resolves its forward references; the hook skips a
  class whose `__pydantic_fields_complete__` is false). Pydantic model-building and
  unresolved-annotation errors propagate unchanged; `model_validate_hdf5` calls
  `model_rebuild()` on an incomplete class so the missing name is reported.
- `_mark` is the **only** function that interprets a child annotation. It accepts a small
  grammar (`Group`, `Dataset`, `np.ndarray`, a jaxtyping array; optionally `| None` or
  `Annotated`; or `dict[str, …]`/`Mapping[str, …]` of those), raises `SchemaError` for
  anything else, and returns the annotation with array leaves wrapped in `Annotated` markers
  (`_NdarraySchema`, `_JaxtypingSchema`), which the hook stores back into `model_fields`.
  Neither `np.ndarray` nor `jaxtyping.AbstractArray` is monkey-patched.
- Attribute models (`T`) are the user's own `BaseModel`s and are **never modified**.
  `_check_attrs` only rejects what cannot be stored (duplicate HDF5 names, bad aliases,
  `Group`/`Dataset` fields, `extra="allow"`). A jaxtyping or `np.ndarray` attribute field is
  validated by Pydantic's plain `isinstance` check, so the model needs
  `arbitrary_types_allowed` and a scalar attribute is not coerced to a 0-d array.
- `localns=` is removed. Resolve local dependencies explicitly with
  `Model.model_rebuild(_types_namespace={...})`; do not inspect frames or clear namespaces.
- The writer checks one thing at write time: the runtime class of an attributes instance
  (`_check_attrs`), since it may be a model no group declared.

### Naming

Internal state is prefixed `_h5t_`. Pydantic excludes annotations starting with `_` from
fields. Group fields represent child HDF5 nodes (attributes live on `.attrs: T`); only repeated
*HDF5* names within a group or attributes model are rejected.

### jaxtyping arrays

jaxtyping does the checking; h5t never parses a shape string. `_mark` accepts an `AbstractArray`
subclass whose `array_type` is `np.ndarray` or a `Dataset` class and rejects any other. The
Pydantic core-schema hook (`_JaxtypingSchema`) calls the non-public
`hint.__instancecheck_str__` on a metadata-only candidate (a zero-stride `_ghost` array, or a
`Dataset`), so no data is read. A guard test fails if jaxtyping drops that method. Dimension
bindings are scoped per group: `Group`'s wrap validator (`_h5t_load`) opens one `jaxtyped("context")` for the
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
  own jaxtyping scope via `handler.generate_schema(attrs_type)`). An in-memory value must
  already be an instance of the declared class: `Dataset[A]` does not coerce an untyped
  `Dataset`.
- jaxtyping arrays and plain `np.ndarray` fields (including collections and chained PEP 695
  aliases of them) read through the `Annotated` markers `_mark` attached, so `read` or
  `_read_array` runs via Pydantic's standard `Annotated` hook without walking or rewriting raw
  `CoreSchema` dicts.
- The `attrs` field validator (`Group._h5t_attrs_scope`) freezes untyped attrs into a
  `_FrozenAttrs` (`_h5view.py`: a read-only `dict` subclass, so it copies, pickles and dumps,
  which `MappingProxyType` does not), whether loaded or built in memory. An untyped `Dataset`
  uses the same class. `extra="forbid"` on a group concerns
  child nodes only.

Pydantic's semantics apply: `T | None` without a default is required, defaults are not
re-validated unless `validate_default`, and under `extra="ignore"` only declared names are read.
Pydantic's `ValidationError` is a `ValueError`, so anything raised inside a validator must be an
`h5t.ValidationError` (not a `ValueError`) to propagate unchanged.

### Error paths are built where the path is known

No code maps a Pydantic `loc` back through the model. Each node that validates raw HDF5
input knows its own absolute path and converts a Pydantic error there with `_located_error`:
`Group._h5t_load` (from `values.node.name`; `loc` parts are child names, or `ATTRS_KEY` then
an attribute name) and the typed `Dataset` validator (from `value.name`, for its attributes).
The result is an `h5t.ValidationError`, which is not a `ValueError`, so it passes through
every enclosing validator unchanged. Messages keep Pydantic's wording (`/values: Field
required`, `/@count: ...`, `/runs/r1/values: ...`); only the first error is reported.
In-memory construction is untouched and raises Pydantic's own `ValidationError`.
Hierarchy: `H5TError` → `SchemaError`, `ValidationError`.

### Writing: trust construction, one pass

`Group.model_dump_hdf5` does not re-validate. Field types, jaxtyping shapes and attribute models
were checked when the instance was built, so `_write._write_group` walks runtime `model_fields`
and creates nodes directly. Typed attributes come from
`model_dump(mode="python", by_alias=True, include=<declared fields>)`, so serializers and
aliases are Pydantic's; nullable field annotations determine which attributes holding `None`
are left out (a required `Any` attribute holding `None` is an error, not silently skipped). The
writer still checks what Pydantic cannot know: collection keys and values h5py cannot store.
An instance built with `model_construct` or mutated after construction is written as it is.

There is no plan phase and no rollback: a failure leaves the nodes already written, and
`mode="w"` has truncated the old file by then. The one thing done before the target is opened
is reading (and caching) lazy `Dataset`s that live in the target file itself, so a
load-then-dump to the same path does not read from a truncated file. Other lazy datasets are
read one at a time and not cached.

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
