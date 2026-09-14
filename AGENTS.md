# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

## Commands

```bash
uv sync                                  # install deps into .venv (uv sync --locked in CI)
uv run pytest                            # full test suite
uv run pytest tests/test_compile.py::test_every_field_kind_and_markers_compile  # single test
uv run --python 3.14 pytest              # PEP 649 paths; skipped on the default interpreter
uv run ty check                          # type check (Astral's ty, not mypy/pyright)
uv run ruff check .                      # lint
uv run ruff format --check src tests     # format check, as CI runs it
```

`[tool.ty.src]` restricts type checking to `src/h5t` and `tests/acceptance_example.py`; other
test modules are deliberately unchecked. `prek.toml` configures ruff-check, ruff-format, and ty
as pre-commit hooks (run via `prek`). Commits follow Conventional Commits (`feat:`, `fix:`,
`chore:`, `refactor!:`).

## Architecture

`h5t` turns annotated Python classes into read-only loaders for HDF5 groups. The load produces
**detached** records: `h5t.load()` closes every handle before returning, and only a `LazyArray`
payload's filename/path survive so its payload can be read later. A schema is always a plain
class decorated with `@h5t.dataset`/`@h5t.group` (or, for a type you cannot decorate, reached
through a `Payload` marker at the use site). h5t contributes no base class and builds no
instance itself -- every record is constructed by calling the record type's own constructor.

Four modules, one direction of dependency (`_cli` → `_compile` → `_array`/`_spec`/`_errors`):

- **`_spec.py`** — the public annotation markers (`Name`, `Attr`, `Eager`, `Payload`) and the
  compiled IR: `FieldSpec` (one field: HDF5 name, `MemberKind`, cached Pydantic `TypeAdapter`,
  default, default factory, converter, an optional `ForeignSpec`) and `ClassSpec` (a record's
  fields + `Extras` policy). `ForeignSpec` is the compiled IR for one record type: the record
  type, its own `ClassSpec`, its `RecordKind`, which of its fields holds the payload
  (`DATASET`-kind only), the declared `LazyArray` type when that payload is lazy, which field (if
  any) receives the attrs snapshot, and the constructor signature when one is available. Also the
  path formatters `child_path` / `attr_path` (`/group/child`, `/group@attr`) used in every error.
  `ClassSpec` survives as a distinct type only because `_check_group_extras`/`_check_dataset_extras`
  take it; folding it into `ForeignSpec` would be churn for no gain.
- **`_array.py`** — `LazyArray`, the detached, lazily-read dataset payload (filename/path/shape/
  dtype snapshot, `.data`/`.read()`/`.open()`). It is reachable two ways: as a record's `Payload`
  field, and as a member annotation in its own right (`samples: h5t.LazyArray`) for a child
  dataset whose attributes the schema does not declare.
- **`_compile.py`** — both halves of the system: annotation → `FieldSpec` compilation, and the
  loader that walks `h5py` nodes producing instances. `MemberKind` (ATTRIBUTE / ARRAY / DATASET /
  GROUP) is the switch that decides which loading path a field takes in `_load_group_values`.
  Within `MemberKind.DATASET`, `field.foreign is not None` is the finer-grained switch between
  `_load_foreign_dataset` (a record type, constructed via `record_type(**values)` through the
  shared `_construct_record`) and constructing `field.member_type` — a `LazyArray` subclass —
  directly. A `GROUP`-kind record field stays `MemberKind.GROUP` and has its `ForeignSpec`
  resolved at load time via `_compiled_record`. `h5t.load(schema, path, root)` is the
  module's public entry point — it resolves and kind-checks `schema` before opening the file, and
  `h5t check` is a thin wrapper around it.
- **`_cli.py`** — `h5t check`, a thin wrapper: import `pkg.mod:Class`, call `h5t.load`, map
  outcomes to exit codes 0 (ok) / 1 (`ValidationError`) / 2 (import, schema, usage, or I/O error).

`RecordKind` (DATASET vs GROUP) is load-bearing and does **not** collapse now that inheritance is
gone: it drives `_field_spec`'s dispatch, `_compile_foreign_fields`'s attributes-only rule, and
`_foreign_spec`'s eager-vs-defer policy.

### Compilation is eager, with a fallback for forward references

The `@h5t.dataset` / `@h5t.group` decorators call `_foreign_spec` at the decorator call, passing
the defining frame's live locals (`inspect.currentframe().f_back.f_locals`) as the resolution
scope. Consequences to keep in mind:

- A `SchemaError` for a bad declaration surfaces from the decorator call, so tests assert it by
  wrapping the decorated `class` statement itself — nothing needs to read the compiled spec.
- A `NameError` raised by the annotation expression itself is *not* an error, **for a
  `GROUP`-kind record**: `_resolved_annotations` raises the private `_UnresolvedAnnotation`,
  `_foreign_spec` lets it through, and the `group()` decorator stores a `_PendingRecord` (carrying
  a weak snapshot of the scope) on `cls.__h5t_record__` instead of a `ForeignSpec`. Only that
  path retains a scope. `@h5t.dataset` does not defer and does not need to — see "Why `DATASET`
  stays eager and `GROUP` defers" below.
- A `NameError` escaping a *function the annotation calls* is a bug in that function, not a
  forward reference, and takes the `SchemaError` path. `_raised_by_annotation` tells the two apart
  by the frame that raised, which it recognises two ways: a PEP 563 string is compiled under the
  `_ANNOTATION_FILE` filename rather than `eval`'s default `<string>`, and a PEP 649 annotation
  runs in the class's own `__annotate__` (matched by code-object identity, since it is compiled
  under the *user's* module filename). A callee's frame is neither. One accepted asymmetry: a
  `NameError` from a lambda *nested inside* an annotation defers on ≤3.13 but is a `SchemaError`
  on 3.14, because the innermost frame is the lambda rather than `__annotate__`.
- On 3.14+, `inspect.get_annotations` is itself the evaluation point (PEP 649), so it raises where
  older versions raised at the `class` statement. `_resolved_annotations` therefore guards that
  call too and routes the failure through the same `_annotation_failure` classifier as the string
  path — a `NameError` from the annotation defers, anything else becomes `SchemaError`. Because
  3.14 does not cache `__annotations__` on the class, a deferred record re-evaluates its annotation
  expressions on the retry: an annotation with side effects runs them twice.
- `_ensure_record_compiled` is the retry point for compile-time callers, and the only way anything
  reads a record's spec without converting a still-unresolved name. It returns a compiled
  `ForeignSpec` unchanged, or retries a `_PendingRecord` against its unpacked scope and caches the
  result back onto `__h5t_record__`. It is called from `_field_spec` (a `DATASET`-kind member).
  Load-time callers (`_load_group_values` for a `GROUP`-kind member, and `h5t.load` itself before
  touching the filesystem) go through `_compiled_record`, which wraps it and turns a renewed
  `_UnresolvedAnnotation` into the user-visible `SchemaError`. Because every read funnels through
  these two, deferral needs no `model_rebuild()`-style public API.
- `_resolved_annotations` searches module globals → scope (live frame locals, or the unpacked
  `_PendingRecord.scope`) → class dict. `_weak_scope`/`_unpack_scope` mirror pydantic's
  `build_lenient_weakvaluedict`: values that reject `weakref.ref` (`int`, `str`, tuples, dicts) are
  stored directly. Because those entries are strong, the snapshot keeps only the names
  `_record_annotation_names` finds in the unresolved annotations — otherwise an unrelated local
  container would pin its contents for the record's lifetime. `_record_annotation_names` unions
  `_annotation_names` across the whole MRO, since a record's bases never compile and snapshot a
  scope of their own.
- `_annotation_names` picks its source from how the annotations failed. If they read back as PEP
  563 strings, `_referenced_names` compiles each and walks it. On 3.14 an unresolved annotation
  never became a value, so the names come off the `__annotate__` code object instead: `_code_names`
  walks `co_names`, and `_quoted_names` recovers the names inside string constants, since a quoted
  annotation is a bare constant that `co_names` never mentions. Both recurse into nested code
  objects and both over-approximate, which is the safe direction. `co_freevars` is deliberately
  *not* collected: an `__annotate__` has free variables only when an enclosing function scope
  exists, which is precisely when its own closure keeps them reachable — the snapshot would be
  redundant there, and collecting them would only widen what `_weak_scope` pins strongly. `annotationlib`'s
  `Format.STRING` and `Format.FORWARDREF` look like the obvious source here and must not be used:
  both first probe the annotate function with the *real* globals and swallow whatever it raises, so
  they would execute a user's annotation helper — and hide its bug — from inside an exception
  handler, breaking the callee-`NameError` rule above.
- Accepted limitation, ≤3.13 and quoted annotations only: a record that both lives in a function
  scope *and* forward-references a class from that same scope may find the referent collected. On
  3.14 a *bare* annotation escapes it entirely — `__annotate__` closes over the defining function's
  cells, so the referent is reachable with or without the snapshot. The flip side is the warning
  worth knowing: that closure makes the class **strongly** retain every enclosing-function local
  its annotations name, deferred or not. That is the retention `_weak_scope` was built to avoid,
  and it is outside h5t's control.

### All internal state is prefixed `_h5t_`, and records have no reserved names

`_compile_foreign_fields` skips any annotation starting with `_`, so h5t's own bookkeeping can
never be mistaken for a field. The one name h5t writes onto a user's class is `__h5t_record__`
(set by the decorator via `setattr`, deliberately not inherited, so a subclass of a decorated
record is not accidentally one itself); `LazyArray`'s cache is `_h5t_data`.

There is no reserved-name check. `dir()` on a plain class yields only its own field names, so a
record field may be called `data`, `attrs`, `path` or `shape` freely — h5t contributes no base
class whose public surface a field name could shadow. Only `_check_duplicate_names` (repeated
*HDF5* names within one record) applies.

### Array annotations see through type aliases

`_is_ndarray_annotation` decides the `ARRAY` kind, and it cannot just read
`typing.get_origin`. numpy ≥ 2.5 defines `npt.NDArray` as a PEP 695 `TypeAliasType`, which defers
its right-hand side, so `get_origin(npt.NDArray[np.float64])` returns the **alias**, not
`np.ndarray`; numpy ≤ 2.4 built the same name by eager substitution, leaving `np.ndarray` directly
visible. The helper expands `__value__` until it reaches a real origin, which also covers a user's
own `type Coords = npt.NDArray[...]` and aliases of aliases. The walk is depth-bounded because a
PEP 695 alias may be self-referential, and `_TYPE_ALIAS_TYPES` is an empty tuple below 3.12 so
every `isinstance` is a cheap `False` there.

This matters across the support matrix rather than within one version: numpy 2.5 requires Python
≥ 3.12, so `uv.lock` resolves 2.4.6 for 3.11 and 2.5.2 for 3.12+. A change here looks
Python-version-dependent in CI but is really numpy-version-dependent.

### Validation

Pydantic is used only through `TypeAdapter` (never `BaseModel`), built once per field at
compile time with `arbitrary_types_allowed=True` so `np.ndarray` passes. Ordering for an
attribute is: raw h5py value → `Attr(converter=...)` (failure ⇒ `ConversionError`) → adapter
(failure ⇒ `ValidationError`). Errors are fail-fast and path-aware; `_short_pydantic_error`
collapses Pydantic's multi-line output to one line. Error hierarchy: `H5TError` → `SchemaError`
(bad declaration) and `ValidationError` (file/schema mismatch) → `ConversionError`.

Missing members go through `_missing_value`: a declared default is itself validated (a
`default_factory` is called first), `T | None` yields `None`, otherwise it is a `ValidationError`.
Either way `_load_attribute` records the value in the `attrs` snapshot, so a field bound with
`attrs=` always contains every declared attribute name.

### Record types (`@h5t.dataset`, `@h5t.group`, `Payload`)

Every schema is a plain record type — typically a stdlib `dataclass` — reached either through a
decorator on the type itself or, for a third-party type you cannot decorate, through a `Payload`
marker at the use site. `RecordKind` (`_spec.py`) tells the two shapes apart: `DATASET` (a
`Payload` field or `@h5t.dataset`) names one field as the payload and may declare nothing but
attributes besides; `GROUP` (`@h5t.group`) has no payload field and instead admits nested groups,
datasets, and arrays. `ForeignSpec.data` is `str | None` because only a `DATASET`-kind record has
one. The word *foreign* survives in the internal names (`_foreign_spec`, `ForeignSpec`,
`_load_foreign_group`) from when these types were the alternative to inheriting from h5t; they are
now simply *the* record types.

`_foreign_spec` compiles a foreign type once per `(kind, data, attrs, extras)` by delegating to
`_compile_foreign_fields`. The outer cache has weak record-type keys and each per-type cache has
weak `ForeignSpec` values: a live decorated marker or owner `FieldSpec` keeps a spec reusable,
without the cached spec's `record_type` back-reference pinning an otherwise transient class. The
compiler walks the type's MRO collecting annotations and their declaring classes, resolves the
payload (`DATASET` only) and `attrs=` fields, then builds every remaining field via
`_field_spec(..., dataset_owner=(kind is RecordKind.DATASET))`. `dataset_owner=True` is what
forces a `DATASET`-kind record to declare only attributes (`_field_spec`'s `"a dataset record may
declare only attributes"` check); a `GROUP`-kind record skips it, since it may reference other
schema types.

**Why `DATASET` stays eager and `GROUP` defers.** A `DATASET`-kind record can only declare
attributes, so it can never forward-reference another schema type; compiling it immediately is
always safe and only improves error locality, so `_foreign_spec` converts any
`_UnresolvedAnnotation` straight into a `SchemaError`. A `GROUP`-kind record has no such
restriction — including a self-reference (`child: Node | None`) — so it needs a "declared before
its dependency" tolerance: `_foreign_spec` lets `_UnresolvedAnnotation` propagate uncaught for
`kind is RecordKind.GROUP`, and the `@h5t.group` decorator (below) catches it there to store a
`_PendingRecord` instead of a `ForeignSpec`. `_record_kind(cls)` reads `.kind` directly off
whichever of the two sits on `cls.__dict__["__h5t_record__"]`, so the DATASET-vs-GROUP dispatch in
`_field_spec` needs no compilation at all — only `_ensure_record_compiled` (retrying a pending
record against its snapshotted scope, and caching the result back onto `__h5t_record__`) forces
that.

**Why a `GROUP`-kind member resolves at load time, not compile time.** `_field_spec`'s `is_record`
branch calls `_ensure_record_compiled` for a `DATASET`-kind member (safe: never pending, never
cyclic) but leaves `FieldSpec.foreign` as `None` for a `GROUP`-kind one, setting only
`kind=MemberKind.GROUP` and storing the class in `FieldSpec.member_type`. `_load_group_values`'s
GROUP branch calls `_compiled_record` on it at load time instead. Eagerly resolving a
`GROUP`-kind member's `ForeignSpec` at compile time would recurse forever on a self-referential
schema like `child: Node | None`, since compiling `Node` would try to compile `Node` again.
Deferring the resolution to load time (once real data bounds the recursion) is what lets it
terminate.

A `MemberKind.DATASET` field with `foreign is None` is the third case: a bare `LazyArray` member,
where `FieldSpec.member_type` is the `LazyArray` subclass and `_load_group_values` constructs it
directly. There is no record type and no attrs snapshot, so `Eager()` is the only marker that
means anything on it.

Four more things fall out of how pydantic and dataclasses actually behave, verified against the
pinned versions in `.venv`:

1. `TypeAdapter(SomeDataclass, config=ConfigDict(...))` raises `PydanticUserError` — pydantic
   rejects `config=` alongside a dataclass/BaseModel/TypedDict type — and a full unconfigured
   adapter would revalidate the record's fields. `_build_record_adapter` instead uses an
   unconfigured `InstanceOf` adapter (or `dict` for a TypedDict runtime value), preserving
   optionality. It accepts constructed records unchanged while still rejecting a wrong-typed
   class-body default or default-factory result. The record type itself remains
   `FieldSpec.annotation` for repr/equivalence purposes.
2. `dataclasses.field(default_factory=...)` leaves **no** class attribute, so reading defaults off
   `cls.__dict__` would silently report such a field as required.
   `_compile_foreign_fields` reads defaults from `dataclasses.fields()` instead when
   the record type is a dataclass, carrying a `default_factory` separately on `FieldSpec` and
   calling it from `_missing_value` before validation. For a non-dataclass it reads a default from
   the class that declared the final annotation, then falls back to a concrete default exposed by
   the inspected constructor signature; the latter covers model frameworks that remove field
   defaults from the class dictionary.
3. `dir()` on a plain dataclass yields only its own field names, so there is no reserved-name
   check to run — only `_check_duplicate_names` (repeated *HDF5* names within one record) applies.
4. `object.__new__` + `object.__setattr__` works even on a `frozen=True, slots=True` dataclass, but
   skips `__init__`/`__post_init__` — wrong for a type that is the user's own. `_load_foreign_dataset`
   and `_load_foreign_group` both call the real constructor through the shared `_construct_record`
   (`record_type(**values)`). It pre-binds the cached signature, making only an argument-binding
   `TypeError` a `SchemaError`; failures raised after user code starts, including a `TypeError` from
   `__post_init__`, become `ValidationError` at the node's path. For an uninspectable callable, the
   traceback's presence or absence of a constructor frame is the fallback distinction.

A record's annotations resolve only against its module globals and class dict by default —
`_resolved_annotations(base)` is called with no `scope`. An `Annotated[T, Payload(...)]` use site
relies on exactly that default; the `@h5t.dataset`/`@h5t.group` decorators (below) are the callers
that pass one, since they do have a defining frame to capture.

### The `@h5t.dataset` / `@h5t.group` decorators

`h5t.dataset(data=..., attrs=..., extras=...)` is sugar for `Annotated[T, Payload(...)]` that
validates against the record type's own definition instead of a distant owner field's. It calls
`_foreign_spec(cls, cls, data, kind=RecordKind.DATASET, data=data, attrs=attrs, extras_raw=extras,
scope=...)` immediately — `cls` is both the record type being compiled and the `owner` a
`SchemaError` names, so a bad `data=` surfaces at the decorator's own call site. Passing the frame
it captures (`inspect.currentframe().f_back.f_locals`) as `_foreign_spec`'s `scope` parameter lets
a function-local record's annotations resolve against function locals, not just module globals. This scope is never retained as a weak snapshot for a later retry — it does
not need to be, since a `DATASET`-kind `_foreign_spec` call always resolves-or-`SchemaError`s
immediately (see above). `_field_spec` dispatches to a decorated record's compiled spec by checking
`"__h5t_record__" in core.__dict__` (a per-class marker set by the decorator's `setattr`,
deliberately not inherited, so a subclass of a decorated record is not accidentally one itself)
rather than `getattr`, and only when no explicit `Payload(...)` is present at the use site — an
explicit marker there still overrides the decorator's own `data=`/`attrs=`/`extras=`, even for a
`GROUP`-kind record (compiling a separate, throwaway `DATASET`-kind spec for it, which then fails
`_field_spec`'s attributes-only check unless the record genuinely has no other child members).

`h5t.group(attrs=..., extras=...)` is the `GROUP`-kind counterpart, with no `data=` since a group
record has no single payload field. It captures the same defining frame, but — unlike `dataset()` —
*does* retain it: on `_UnresolvedAnnotation` it stores a `_PendingRecord` (kind, data, attrs,
extras_raw, py_name, and `_weak_scope(scope, _record_annotation_names(cls))`) on
`cls.__h5t_record__` instead of a `ForeignSpec`. `_record_annotation_names` unions
`_annotation_names` across the record's whole MRO rather than reading `cls` alone, since a
record's bases never compile and snapshot a scope of their own. `_ensure_record_compiled` is the
retry point: called from `_field_spec` (a `DATASET`-kind member) it returns a compiled
`ForeignSpec` unchanged or retries a `_PendingRecord` against its unpacked scope, letting a
renewed `_UnresolvedAnnotation` propagate uncaught so that a `GROUP`-kind owner record's own
`_ensure_record_compiled` retry defers too. Load-time callers — `_load_group_values` for a nested
`GROUP` member, and `h5t.load` for a top-level record — go through `_compiled_record`, which
converts that renewed exception into `SchemaError`. `h5t check` only catches `SchemaError`, so
leaving the private exception uncaught there would surface as an internal traceback.

## Testing conventions

`tests/conftest.py` holds the canonical schemas: `Result` (a `@h5t.group` record exercising every
field kind) with its nested `Nested`, plus `EagerMeasurement` (decorated `@h5t.dataset` over an
`np.ndarray` payload, so it also serves as a dataset-record CLI/`h5t.load` fixture) and
`LazyMeasurement` (undecorated, reached through the `Payload` marker, `LazyArray` payload) over
the same `measurement` dataset. `write_result()` writes a matching file including undeclared
members, to exercise `extras`; `open_fd_count()` (POSIX-only, reads `/dev/fd`) is shared by every
fd-leak assertion. Prefer extending those over new ad-hoc fixtures.

Module split: `tests/test_compile.py` covers field classification, deferral and declaration
errors; `tests/test_foreign.py` covers the `Payload` marker specifically; `tests/test_records.py`
covers the `@h5t.dataset`/`@h5t.group` decorator API — including forward-reference and
self-referential (`GROUP`-kind) schemas, and bare `LazyArray` members.

`tests/test_pep649.py` is the one module deliberately *without* `from __future__ import
annotations` — it exists to exercise the lazy-annotation paths every other module opts out of, and
adding a future import there would silently void the whole file (hence
`test_this_module_keeps_lazy_annotations`). It skips below 3.14, so `.python-version`'s 3.11 never
runs it; use `uv run --python 3.14 pytest`. Its record types are declared *inside* the test
functions, since a module-level bare forward reference would raise during collection on the
versions the skip covers. Note the asymmetry it pins: `@h5t.dataset` does not defer even under
PEP 649 (attributes-only, so nothing is gained by deferring), while `@h5t.group` does.

`tests/acceptance_example.py` is not collected by pytest — it is a static acceptance surface
whose annotated assignments (`version: int = result.version`) fail `ty check` if `h5t.load`
stops returning precisely-typed records. Update it when the public typing surface changes.

`tests/test_dataset.py` asserts no file descriptors leak using `open_fd_count()` (POSIX-only,
reads `/dev/fd`); any change to handle lifetime in `h5t.load` or `LazyArray.open` must keep those
green.
