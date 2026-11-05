# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Documentation

- Sphinx documentation site (getting started, user guide, API reference) for Read the Docs.
- Numpy-style docstrings for the whole public API.
- README trimmed to a short pitch and one example; the rest moved into the user guide.

## [0.4.0]

### Added

- `h5t.save(record, path, root="/", mode="w-", overwrite=False)`: a write path driven by
  the same schema. Records are validated (including jaxtyping shapes and dtypes) and every
  lazy payload is read before the file is opened; a failed write removes what it wrote
  (best effort for `mode="a"`). `Attr(serializer=...)` is the inverse of `converter`.
  `FieldSpec` (internal) gains a `serializer` field.
- Dynamic collections: `dict[str, T]` / `Mapping[str, T]` fields load every child of one
  named group into a `dict`, and the new `h5t.Glob(pattern)` marker filters the children.
- jaxtyping array validation: `Float[np.ndarray, "n n"]` and `Float[h5t.LazyArray, "n _"]`
  fields (and dataset-record payloads) have their dtype and shape checked by jaxtyping
  against the file's metadata, with no data read and no casting. Named dimensions unify
  across the array fields of one record. Plain `np.ndarray` and `NDArray[...]` stay
  unchecked.
- `localns=` on `@h5t.dataset` and `@h5t.group`: extra names for resolving the record's
  annotations.

### Changed

- **Behavior change:** a deferred `@h5t.group` record no longer retains a snapshot of the
  decorating function's locals. A function-local record that defers on a later-defined
  name *and* uses a function-local type now needs `localns={...}`; otherwise `h5t.load`
  raises `SchemaError`. Function-local self-references now resolve.
- **Behavior change:** annotation resolution uses `typing.get_type_hints` (<=3.13) /
  `annotationlib` (3.14+) instead of h5t's own bytecode scanning. A broken annotation on a
  base class now fails the subclass, and names in the class body no longer resolve in
  string annotations.
- **New dependency:** `jaxtyping>=0.2.36` is now a required runtime dependency.

## [0.3.0]

### Removed

- **Breaking:** the `Group`/`Dataset` inheritance API is gone. `h5t.Group`, `h5t.Dataset`,
  `Group.from_file()`, the `extras=` class keyword, and the reserved-name check that
  forbade a field shadowing a `Group`/`Dataset` attribute (`data`, `attrs`, `path`, ...)
  no longer exist. `@h5t.dataset`/`@h5t.group` plus `h5t.load()` already covered every
  position a schema can occupy -- root, nested group, nested dataset -- so inheritance
  retained no capability of its own. To migrate:

  | Before | After |
  | --- | --- |
  | `class X(h5t.Group, extras="forbid")` | `@h5t.group(extras="forbid")` over a `@dataclass` |
  | `class X(h5t.Dataset)` | `@h5t.dataset(data="...")` over a `@dataclass` |
  | `x: h5t.Dataset` | `x: h5t.LazyArray` |
  | `X.from_file(path, root)` | `h5t.load(X, path, root)` |
  | `dataset.attrs` | a `Mapping` field bound with `attrs="..."` |
  | `dataset.data` / `.read()` / `.open()` / `.shape` | the same names on the record's `LazyArray` payload |

  A record is now always built by calling the record type's own constructor, so a loaded
  record has whatever `__init__`/`__post_init__`/`__eq__` its type defines.

### Added

- `LazyArray` (or a subclass of it) is accepted as a member annotation in its own right,
  for a child dataset whose attributes the schema does not declare. `Eager()` still
  prefetches such a field's payload during the load.

## [0.2.0]

### Changed

- **Breaking:** schemas now compile at the `class` statement instead of on first use. A bad
  declaration raises `SchemaError` from the `class` statement itself; a class that forward-
  references a name not yet defined in its module defers compilation until first use instead.
- **Breaking:** loaded records are now detached, read-only snapshots rather than live views over
  an open file. `Group.from_file()` closes every handle before returning; only a `Dataset`'s
  filename/path survive so its payload can be read later. `Group`/`Dataset` no longer have a
  working constructor, `__eq__`, or serialization.

### Added

- Node validators replace the previous shape-constraint annotations for validating array fields.
- `ty` replaces mypy/pyright for type checking, with `prek` hooks for ruff and ty.

## [0.1.0]

- Initial release: `Group`/`Dataset` schema classes, `Name`/`Attr`/`Eager` markers, the
  `h5t check` CLI, and Pydantic-backed attribute validation.

[Unreleased]: https://github.com/binado/h5t/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/binado/h5t/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/binado/h5t/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/binado/h5t/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/binado/h5t/commits/v0.1.0
