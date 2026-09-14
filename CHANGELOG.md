# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/binado/h5t/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/binado/h5t/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/binado/h5t/commits/v0.1.0
