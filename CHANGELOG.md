# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
