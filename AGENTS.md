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
**detached** records: `h5t.load()` closes every handle before returning, and only a `Dataset`'s
(or a `LazyArray` payload's) filename/path survive so its payload can be read later. Plain classes
decorated with `@h5t.dataset`/`@h5t.group` are the primary API; `Group`/`Dataset` inheritance is
legacy (see `README.md`) and exists mainly so the two styles can freely mix within one schema.

Four modules, one direction of dependency (`_cli` → `_compile` → `_array`/`_spec`/`_errors`):

- **`_spec.py`** — the public annotation markers (`Name`, `Attr`, `Eager`, `Payload`) and the
  compiled IR: `FieldSpec` (one field: HDF5 name, `MemberKind`, cached Pydantic `TypeAdapter`,
  default, default factory, converter, an optional `ForeignSpec`) and `ClassSpec` (a class's
  flattened fields + `Extras` policy). `ForeignSpec` is the compiled IR for a `Payload` field: the
  foreign record type, its own `ClassSpec`, which of its fields holds the payload, whether that
  payload is a `LazyArray`, which field (if any) receives the attrs snapshot, and the inspected
  constructor signature when one is available. Also the path formatters `child_path` / `attr_path`
  (`/group/child`, `/group@attr`) used in every error.
- **`_array.py`** — `LazyArray`, the detached, lazily-read dataset payload (filename/path/shape/
  dtype snapshot, `.data`/`.read()`/`.open()`). `Dataset` delegates to an internal `_h5t_array:
  LazyArray` rather than duplicating this state; a `Payload` field with a `LazyArray`-typed member
  uses the same class directly.
- **`_compile.py`** — both halves of the system: annotation → `FieldSpec` compilation, and the
  loader that walks `h5py` nodes producing instances. `MemberKind` (ATTRIBUTE / ARRAY / DATASET /
  GROUP) is the switch that decides which loading path a field takes in `_load_group_values`
  (the body shared by `_load_group` and `_load_foreign_group`). A `Payload` field keeps
  `MemberKind.DATASET` — `field.foreign is not None` is the finer-grained switch between
  `_load_dataset` (an `h5t.Dataset` subclass) and `_load_foreign_dataset` (a plain record type,
  constructed via `record_type(**values)` through the shared `_construct_record`). A `GROUP`-kind
  record field instead stays `MemberKind.GROUP` and is told apart from a plain `Group` subclass
  field at load time, by `"__h5t_record__" in field.member_type.__dict__`. `h5t.load(schema, path,
  root)` is the module's public entry point — it dispatches on `schema` (a `Group` subclass or a
  decorated record) before opening the file, and both `Group.from_file` and `h5t check` are thin
  wrappers around it.
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
`tests/test_loading.py` and `tests/test_compile.py::test_removed_api_is_absent`. A decorated record
is the exception to "never constructed", in both its kinds: `_load_foreign_dataset` (a `Payload`
field, `@h5t.dataset`) and `_load_foreign_group` (`@h5t.group`) both call `_construct_record`, which
calls the foreign record type's real `record_type(**values)`, because that type is the user's own
and is expected to run its own `__init__`/`__post_init__`.

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
`data`, `attrs`, `lazy`, extras policy) for a `Payload` field. `extras` is inherited from the
nearest compiled base unless the subclass passes it explicitly.

### Foreign record types (`Payload`, `@h5t.dataset`, `@h5t.group`)

A child dataset or group does not have to load into an `h5t.Dataset`/`h5t.Group` subclass. A
`Payload` field (or the `@h5t.dataset`/`@h5t.group` decorators, below) loads it into a plain record
type instead — typically a stdlib `dataclass` — so the record does not have to inherit from h5t and
is not subject to `Dataset`'s/`Group`'s reserved-name check. `RecordKind` (`_spec.py`) tells the two
shapes apart: `DATASET` (a `Payload` field or `@h5t.dataset`) names one field as the payload;
`GROUP` (`@h5t.group`) has no payload field and instead admits nested groups, datasets, and arrays
like a `Group` subclass does. `ForeignSpec.data` is `str | None` because only a `DATASET`-kind
record has one.

`_foreign_spec` compiles a foreign type once per `(kind, data, attrs, extras)` by delegating to
`_compile_foreign_fields`. The outer cache has weak record-type keys and each per-type cache has
weak `ForeignSpec` values: a live decorated marker or owner `FieldSpec` keeps a spec reusable,
without the cached spec's `record_type` back-reference pinning an otherwise transient class. The
compiler walks the type's MRO collecting annotations and their declaring classes, resolves the
payload (`DATASET` only) and `attrs=` fields, then builds every remaining field via
`_field_spec(..., dataset_owner=(kind is RecordKind.DATASET))`. `dataset_owner=True` is what still
forces a `DATASET`-kind record to declare only attributes (`_field_spec`'s `"a dataset record may
declare only attributes"` check); a `GROUP`-kind record skips it, since — like `Group` — it may
reference other schema types.

**Why `DATASET` stays eager and `GROUP` defers.** A `DATASET`-kind record can only declare
attributes, so it can never forward-reference another schema type; compiling it immediately is
always safe and only improves error locality, so `_foreign_spec` converts any
`_UnresolvedAnnotation` straight into a `SchemaError`. A `GROUP`-kind record has no such
restriction — including a self-reference (`child: Node | None`) — so it needs the same
"declared before its dependency" tolerance `_Record.__init_subclass__` already gives `Group`/
`Dataset`: `_foreign_spec` lets `_UnresolvedAnnotation` propagate uncaught for `kind is
RecordKind.GROUP`, and the `@h5t.group` decorator (below) catches it there to store a
`_PendingRecord` instead of a `ForeignSpec`. `_record_kind(cls)` reads `.kind` directly off
whichever of the two sits on `cls.__dict__["__h5t_record__"]`, so the DATASET-vs-GROUP dispatch in
`_field_spec` needs no compilation at all — only `_ensure_record_compiled` (retrying a pending
record against its snapshotted scope, and caching the result back onto `__h5t_record__`) forces
that.

**Why a `GROUP`-kind member resolves at load time, not compile time.** `_field_spec`'s `is_record`
branch calls `_ensure_record_compiled` for a `DATASET`-kind member (safe: never pending, never
cyclic) but leaves `FieldSpec.foreign` as `None` for a `GROUP`-kind one, setting only
`kind=MemberKind.GROUP`. This mirrors exactly how a plain `Group`-subclass field already works —
`FieldSpec.member_type` stores the class, and its `__h5spec__` is read lazily by `_load_group_values`
at load time (`:942`-era comment; see below) — and for the same reason: eagerly resolving a
`GROUP`-kind member's `ForeignSpec` at compile time would recurse forever on a self-referential
schema like `child: Node | None`, since compiling `Node` would try to compile `Node` again. Deferring
the resolution to load time (once real data bounds the recursion) is what lets it terminate.

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
   `cls.__dict__` (as `_own_fields` does for h5t's own classes) would silently report such a field
   as required. `_compile_foreign_fields` reads defaults from `dataclasses.fields()` instead when
   the record type is a dataclass, carrying a `default_factory` separately on `FieldSpec` and
   calling it from `_missing_value` before validation. For a non-dataclass it reads a default from
   the class that declared the final annotation, then falls back to a concrete default exposed by
   the inspected constructor signature; the latter covers model frameworks that remove field
   defaults from the class dictionary.
3. `dir()` on a plain dataclass yields only its own field names, so `_check_fields`'s reserved-name
   half has nothing to forbid there — only `_check_duplicate_names` (the half checking for repeated
   HDF5 names, extracted out for reuse) applies to a foreign record's fields.
4. `object.__new__` + `object.__setattr__` works even on a `frozen=True, slots=True` dataclass, but
   skips `__init__`/`__post_init__` — wrong for a type that is the user's own. `_load_foreign_dataset`
   and `_load_foreign_group` both call the real constructor through the shared `_construct_record`
   (`record_type(**values)`). It pre-binds the cached signature, making only an argument-binding
   `TypeError` a `SchemaError`; failures raised after user code starts, including a `TypeError` from
   `__post_init__`, become `ValidationError` at the node's path. For an uninspectable callable, the
   traceback's presence or absence of a constructor frame is the fallback distinction.

Foreign annotations resolve only against the record type's module globals and class dict by
default — `_resolved_annotations(base)` is called with no `scope`. An `Annotated[T, Payload(...)]`
use site relies on exactly that default; the `@h5t.dataset`/`@h5t.group` decorators (below) are the
callers that pass one, since they do have a defining frame to capture.

### The `@h5t.dataset` / `@h5t.group` decorators

`h5t.dataset(data=..., attrs=..., extras=...)` is sugar for `Annotated[T, Payload(...)]` that
validates against the record type's own definition instead of a distant owner field's. It calls
`_foreign_spec(cls, cls, data, kind=RecordKind.DATASET, data=data, attrs=attrs, extras_raw=extras,
scope=...)` immediately — `cls` is both the record type being compiled and the `owner` a
`SchemaError` names, so a bad `data=` surfaces at the decorator's own call site. The frame it
captures (`inspect.currentframe().f_back.f_locals`) is the same one `_Record.__init_subclass__`
would capture were `cls` a `Group`/`Dataset` subclass instead; passing it as `_foreign_spec`'s
`scope` parameter lets a function-local record's annotations resolve against function locals, not
just module globals. This scope is never retained as a weak snapshot for a later retry — it does
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
`cls.__h5t_record__` instead of a `ForeignSpec`. `_record_annotation_names` unions `_annotation_names`
across the record's whole MRO (not just `cls` itself, unlike `_Record`'s single-class version),
since a foreign record's bases never compile and snapshot their own scope the way an `h5t.Group`
base does. `_ensure_record_compiled` is the retry point: called from `_field_spec` (a `DATASET`-kind
member), from `_load_group_values` (a `GROUP`-kind member, at load time), and from `h5t.load`
itself, it returns a compiled `ForeignSpec` unchanged or retries a `_PendingRecord` against its
unpacked scope, letting a renewed `_UnresolvedAnnotation` propagate uncaught so that whatever
compiled `cls` as a field (an owner `Group`/`Dataset`'s own `_compile_class`/`__init_subclass__`, or
a `GROUP`-kind owner record's own `_ensure_record_compiled` retry) defers too, via the machinery
that already exists for exactly this.

## Testing conventions

`tests/conftest.py` holds the canonical `Result` / `Measurement` / `Nested` schemas exercising
every field kind, plus `PlainMeasurement` (decorated `@h5t.dataset`, so it also serves as a
dataset-record CLI/`h5t.load` fixture) / `LazyMeasurement` (foreign records over the same
`measurement` dataset, eager and `LazyArray` payloads respectively), `PlainNested` /
`PlainResult` (`@h5t.group` records mirroring `Nested`/`Result` field-for-field over the same
file), `write_result()` which writes a matching file (including undeclared members, to exercise
`extras`), and `open_fd_count()` (POSIX-only, reads `/dev/fd`) shared by every fd-leak assertion.
Prefer extending those over new ad-hoc fixtures. `tests/test_foreign.py` covers the `Payload`
marker and `LazyArray` specifically; `tests/test_records.py` covers both the `@h5t.dataset` and
`@h5t.group` decorator API, which compile through the same `_foreign_spec`/`ForeignSpec` machinery
— including forward-reference and self-referential (`GROUP`-kind) schemas, mixing decorated
records with `Group`/`Dataset` inheritance in both directions, and `PlainResult`'s field-for-field
parity against `Result`.

`tests/test_pep649.py` is the one module deliberately *without* `from __future__ import
annotations` — it exists to exercise the lazy-annotation paths every other module opts out of, and
adding a future import there would silently void the whole file (hence
`test_this_module_keeps_lazy_annotations`). It skips below 3.14, so `.python-version`'s 3.11 never
runs it; use `uv run --python 3.14 pytest`. Its schema classes are declared *inside* the test
functions, since a module-level bare forward reference would raise during collection on the
versions the skip covers. Note the asymmetry it pins: `@h5t.dataset` still does not defer even
under PEP 649 (attributes-only, so nothing is gained by deferring), while `@h5t.group` does, the
same as `Group`/`Dataset`.

`tests/acceptance_example.py` is not collected by pytest — it is a static acceptance surface
whose annotated assignments (`version: int = result.version`) fail `ty check` if `from_file`
stops returning precisely-typed records. Update it when the public typing surface changes.

`tests/test_dataset.py` asserts no file descriptors leak using `open_fd_count()` (POSIX-only,
reads `/dev/fd`); any change to handle lifetime in `from_file` or `Dataset.open` must keep those
green.
