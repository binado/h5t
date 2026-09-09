# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

## Commands

```bash
uv sync                                  # install deps into .venv (uv sync --locked in CI)
uv run pytest                            # full test suite
uv run pytest tests/test_compile.py::test_inheritance_defaults_and_extras   # single test
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
**detached** records: `Group.from_file()` closes every handle before returning, and only a
`Dataset`'s filename/path survive so its payload can be read later.

Four modules, one direction of dependency (`_cli` → `_compile` → `_array`/`_spec`/`_errors`):

- **`_spec.py`** — the public annotation markers (`Name`, `Attr`, `Eager`, `Payload`) and the
  compiled IR: `FieldSpec` (one field: HDF5 name, `MemberKind`, cached Pydantic `TypeAdapter`,
  default, default factory, converter, an optional `ForeignSpec`) and `ClassSpec` (a class's
  flattened fields + `Extras` policy). `ForeignSpec` is the compiled IR for a `Payload` field: the
  foreign record type, its own `ClassSpec`, which of its fields holds the payload, and whether that
  payload is eager or a `LazyArray`. Also the path formatters `child_path` / `attr_path`
  (`/group/child`, `/group@attr`) used in every error.
- **`_array.py`** — `LazyArray`, the detached, lazily-read dataset payload (filename/path/shape/
  dtype snapshot, `.data`/`.read()`/`.open()`). `Dataset` delegates to an internal `_h5t_array:
  LazyArray` rather than duplicating this state; a `Payload` field with a `LazyArray`-typed member
  uses the same class directly.
- **`_compile.py`** — both halves of the system: annotation → `FieldSpec` compilation, and the
  loader that walks `h5py` nodes producing instances. `MemberKind` (ATTRIBUTE / ARRAY / DATASET /
  GROUP) is the switch that decides which loading path a field takes in `_load_group`. A `Payload`
  field keeps `MemberKind.DATASET` — `field.foreign is not None` is the finer-grained switch
  between `_load_dataset` (an `h5t.Dataset` subclass) and `_load_foreign_dataset` (a plain record
  type, constructed via `record_type(**values)`).
- **`_cli.py`** — `h5t check`, a thin wrapper: import `pkg.mod:Class`, call `from_file`, map
  outcomes to exit codes 0 (ok) / 1 (`ValidationError`) / 2 (import, schema, usage, or I/O error).

### Compilation is eager, with a fallback for forward references

`_Record.__init_subclass__` calls `_compile_class` at the `class` statement, passing the defining
frame's live locals as the resolution scope. Consequences to keep in mind:

- A `SchemaError` for a bad declaration surfaces from the `class` statement, so tests assert it by
  wrapping the statement itself — nothing needs to touch `.__h5spec__`.
- A `NameError` raised by the annotation expression itself is *not* an error:
  `_resolved_annotations` raises the private `_UnresolvedAnnotation`, `__init_subclass__` swallows
  it and stores a weak snapshot of the scope in `_h5t_scope`, and the class compiles later. Only
  that path retains a scope. A subclass of a class that deferred defers too (`_compile_class`'s
  base loop lets the exception through), so the whole subtree resolves together.
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
  3.14 does not cache `__annotations__` on the class, a deferred class re-evaluates its annotation
  expressions on the retry: an annotation with side effects runs them twice.
- `SchemaMeta.__h5spec__` is a metaclass *property* whose `_ensure_compiled` is now the fallback
  route rather than the common one; it turns a still-unresolved name into the user-visible
  `SchemaError`. `from_file` triggers it before touching the filesystem. Because every read of the
  spec funnels through it, deferral needs no `model_rebuild()`-style public API.
- `Group` and `Dataset` themselves are excluded by the `cls.__module__ == __name__` guard: while
  their `class` statements run, the module global `Dataset` that `_compile_class` reads is not yet
  bound. They compile lazily, as before.
- `_resolved_annotations` searches module globals → scope (live frame locals, or the unpacked
  `_h5t_scope`) → class dict. `_weak_scope`/`_unpack_scope` mirror pydantic's
  `build_lenient_weakvaluedict`: values that reject `weakref.ref` (`int`, `str`, tuples, dicts) are
  stored directly. Because those entries are strong, the snapshot keeps only the names
  `_annotation_names` finds in the unresolved annotations — otherwise an unrelated local container
  would pin its contents for the class's lifetime.
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
- Accepted limitation, ≤3.13 and quoted annotations only: a schema that both lives in a function
  scope *and* forward-references a class from that same scope may find the referent collected. On
  3.14 a *bare* annotation escapes it entirely — `__annotate__` closes over the defining function's
  cells, so the referent is reachable with or without the snapshot. The flip side is the warning
  worth knowing: that closure makes the class **strongly** retain every enclosing-function local
  its annotations name, deferred or not. That is the retention `_weak_scope` was built to avoid,
  and it is outside h5t's control.

### Instances are built by the loader, never constructed

`Group.__init__` / `Dataset.__init__` raise `TypeError`. `_load_group` / `_load_dataset` use
`object.__new__` plus `object.__setattr__` to populate fields and the `_h5t_*` internals —
`Dataset`'s only such internal is `_h5t_array: LazyArray`, which its properties delegate to. Do not
add a working constructor, `__eq__`, or serialization — their absence is asserted by
`tests/test_loading.py` and `tests/test_compile.py::test_removed_api_is_absent`. A `Payload` field
is the one exception to "never constructed": `_load_foreign_dataset` calls the foreign record
type's real `record_type(**values)`, because that type is the user's own and is expected to run its
own `__init__`/`__post_init__`.

All internal instance state is prefixed `_h5t_`. Two invariants follow from this: `_own_fields`
skips any annotation starting with `_`, and `_check_fields` rejects a field whose Python name
collides with a public attribute of `Group`/`Dataset` (computed live via `dir()`). **Adding a
public method or property to `Group` or `Dataset` therefore breaks user schemas that already use
that field name** — treat the public surface of those two classes as near-frozen. This reserved-name
check does not apply to a `Payload` record type: `dir()` on a plain dataclass yields only its field
names, so there is nothing to forbid, and a field named `data` (reserved on `Dataset`) is legal
there.

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

Missing members go through `_missing_value`: a class-body default is itself validated, `T | None`
yields `None`, otherwise it is a `ValidationError`. Either way `_load_attribute` records the
value in the `attrs` snapshot, so `attrs` always contains every declared attribute name.

### Inheritance

`_merged_fields` walks the reversed MRO collecting each class's `_h5t_own`. A subclass overrides
a base's field; two unrelated bases declaring the same name must be *equivalent* per
`_fields_equivalent`, which compares `repr()` of annotation and default rather than `==` to avoid
ndarray-truthiness ambiguity and adapter identity, plus `_foreign_equivalent` (record type,
`data_attr`, `lazy`, extras policy) for a `Payload` field. `extras` is inherited from the nearest
compiled base unless the subclass passes it explicitly.

### Foreign record types (`Payload`)

A `Payload` field loads a child dataset into a plain record type (typically a stdlib `dataclass`)
instead of an `h5t.Dataset` subclass, so the record does not have to inherit from h5t and is not
subject to `Dataset`'s reserved-name check. `_foreign_spec` compiles the foreign type once per
`(record_type, data_attr, extras)` (cached in a `WeakKeyDictionary`), reusing `_field_spec` with
`dataset_owner=True` for every field but the payload. Four things fall out of how pydantic and
dataclasses actually behave, verified against the pinned versions in `.venv`:

1. `TypeAdapter(SomeDataclass, config=ConfigDict(...))` raises `PydanticUserError` — pydantic
   rejects `config=` alongside a dataclass/BaseModel/TypedDict type — and dropping `config=` buys
   nothing either, since `validate_python(instance)` on an already-built instance is a no-op. A
   `Payload` field therefore gets no member-level adapter for the record type itself: `_field_spec`
   builds it from `Any` (which accepts the constructed instance unchanged) while keeping the record
   type as `FieldSpec.annotation` for repr/equivalence purposes only.
2. `dataclasses.field(default_factory=...)` leaves **no** class attribute, so reading defaults off
   `cls.__dict__` (as `_own_fields` does for h5t's own classes) would silently report such a field
   as required. `_foreign_spec` reads defaults from `dataclasses.fields()` instead when the record
   type is a dataclass, carrying a `default_factory` separately on `FieldSpec` and calling it from
   `_missing_value` before validation.
3. `dir()` on a plain dataclass yields only its own field names, so `_check_fields`'s reserved-name
   half has nothing to forbid there — only `_check_duplicate_names` (the half checking for repeated
   HDF5 names, extracted out for reuse) applies to a foreign record's fields.
4. `object.__new__` + `object.__setattr__` works even on a `frozen=True, slots=True` dataclass, but
   skips `__init__`/`__post_init__` — wrong for a type that is the user's own. `_load_foreign_dataset`
   calls the real constructor, `record_type(**values)`, instead: a `TypeError` (signature mismatch)
   becomes `SchemaError`, any other failure (including a `__post_init__` invariant) becomes
   `ValidationError` at the dataset's path.

Foreign annotations resolve only against the record type's module globals and class dict —
`_resolved_annotations(base)` is called with no `scope`, since there is no `__init_subclass__` hook
on a plain class to capture a defining frame's locals the way `_Record`'s deferred path does. An
unresolved annotation there is therefore always a `SchemaError`, never something the
`_h5t_scope`/`_weak_scope` deferral machinery retries later.

## Testing conventions

`tests/conftest.py` holds the canonical `Result` / `Measurement` / `Nested` schemas exercising
every field kind, plus `PlainMeasurement` / `LazyMeasurement` (foreign records over the same
`measurement` dataset, eager and `LazyArray` payloads respectively) and `write_result()` which
writes a matching file (including undeclared members, to exercise `extras`). Prefer extending
those over new ad-hoc fixtures. `tests/test_foreign.py` covers `Payload`/`LazyArray` specifically.

`tests/test_pep649.py` is the one module deliberately *without* `from __future__ import
annotations` — it exists to exercise the lazy-annotation paths every other module opts out of, and
adding a future import there would silently void the whole file (hence
`test_this_module_keeps_lazy_annotations`). It skips below 3.14, so `.python-version`'s 3.11 never
runs it; use `uv run --python 3.14 pytest`. Its schema classes are declared *inside* the test
functions, since a module-level bare forward reference would raise during collection on the
versions the skip covers.

`tests/acceptance_example.py` is not collected by pytest — it is a static acceptance surface
whose annotated assignments (`version: int = result.version`) fail `ty check` if `from_file`
stops returning precisely-typed records. Update it when the public typing surface changes.

`tests/test_dataset.py` asserts no file descriptors leak using `open_fd_count()` (POSIX-only,
reads `/dev/fd`); any change to handle lifetime in `from_file` or `Dataset.open` must keep those
green.
