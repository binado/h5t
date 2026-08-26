# Detached-model refactor

Version 0.2 is a clean break from h5t's live-view API. The schema class is now a
dataclass-like read model: `Group.from_file()` recursively validates and materializes a
group, then returns only after its HDF5 handle has closed.

## Migration

| Removed 0.1 API | 0.2 replacement |
| --- | --- |
| `class Result(h5t.File)` | `class Result(h5t.Group)` |
| `with Result.open(path) as result` | `result = Result.from_file(path)` |
| `Dataset[h5t.f8]` and dtype tokens | bare `Dataset`, a `Dataset` subclass, or `np.ndarray` |
| dataset slicing through a retained view | `with dataset.open() as live: live[...]` |
| complete dataset read with `dataset[:]` | `dataset.data` or `dataset.read()` |
| `Group[T]` and `Keys(...)` | no replacement yet; dynamic collections are deferred |
| `validate()`, `Invalid`, and `.check()` | field validation through Pydantic while loading |
| aggregate `ValidationReport` | first path-aware `ValidationError` |
| `extras="warn"` | choose `"ignore"` or `"forbid"` on each schema class |
| `validate_schema()` | declarations compile on first use and raise `SchemaError` |
| `.close()` and closed-view errors | detached records need no close operation |

There are intentionally no aliases for removed names. This keeps annotations, runtime
semantics, and static typing aligned with the detached model.

## Consistency boundary

`from_file()` snapshots all group attributes, typed-dataset attributes and metadata,
nested groups, eager datasets, and plain NumPy arrays in one open-file lifetime. Lazy
typed payloads are outside that snapshot: their first `.data` read may see a later file
version, and `.open()` always exposes a fresh live dataset. Once `.data` succeeds, it is
cached and every subsequent `.data` or `.read()` call returns the same ndarray object.

This split supports cheap structural loading while making source-file changes explicit:
metadata can remain available after deletion, whereas an uncached payload or `.open()`
requires the absolute source filename and HDF5 path to still resolve.
