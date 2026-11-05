"""Schema-validated writing of records to HDF5, in two phases.

Phase 1 (*plan*) walks the compiled spec over an in-memory record and builds a small tree
of what to write, validating everything and reading every lazy payload. It never touches
the target file. Phase 2 (*apply*) opens the file and writes the tree, removing what it
wrote if anything fails.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import typing
from collections.abc import Mapping
from enum import Enum
from typing import Any, Literal

import h5py
import numpy as np
from jaxtyping import jaxtyped

from h5t._array import _DATA_NOT_LOADED, LazyArray
from h5t._compile import (
    _check_array,
    _compiled_record,
    _filesystem_path,
    _normalize_root,
    _validate_value,
)
from h5t._errors import ConversionError, SchemaError, ValidationError
from h5t._spec import (
    ArraySpec,
    Extras,
    FieldSpec,
    ForeignSpec,
    MemberKind,
    RecordKind,
    attr_path,
    child_path,
)


@dataclasses.dataclass
class _DatasetPlan:
    data: np.ndarray
    attrs: dict[str, Any]


@dataclasses.dataclass
class _GroupPlan:
    attrs: dict[str, Any] = dataclasses.field(default_factory=dict)
    children: dict[str, _GroupPlan | _DatasetPlan] = dataclasses.field(default_factory=dict)


# ---------------------------------------------------------------------------
# Phase 1: plan
# ---------------------------------------------------------------------------


def _read_field(foreign: ForeignSpec, record: Any, name: str) -> Any:
    try:
        return getattr(record, name)
    except AttributeError as exc:
        raise SchemaError(
            f"{foreign.record_type.__name__}.{name}: record has no such attribute"
        ) from exc


def _check_storable(value: Any, path: str) -> None:
    """Reject values h5py would store as an object array, or fail on."""
    if isinstance(value, (str, bytes)):
        return
    try:
        kind = np.asarray(value).dtype.kind
    except Exception as exc:
        raise ValidationError(path, f"cannot store {type(value).__name__}: {exc}") from exc
    if kind == "O":
        raise ValidationError(
            path,
            f"cannot store {type(value).__name__} as an HDF5 attribute; add Attr(serializer=...)",
        )


def _serialize_attribute(field: FieldSpec, value: Any, path: str) -> Any:
    if field.serializer is not None:
        try:
            value = field.serializer(value)
        except Exception as exc:
            raise ConversionError(path, f"attribute serializer failed: {exc}") from exc
    elif isinstance(value, Enum):
        value = value.value
    _check_storable(value, path)
    return value


def _plan_attributes(
    foreign: ForeignSpec, record: Any, path: str, fields: tuple[FieldSpec, ...]
) -> dict[str, Any]:
    """Validate and serialize ``fields``' attribute members of ``record``."""
    attrs: dict[str, Any] = {}
    for field in fields:
        if field.kind is not MemberKind.ATTRIBUTE:
            continue
        apath = attr_path(path, field.h5_name)
        value = _validate_value(field, _read_field(foreign, record, field.py_name), apath)
        if value is None:
            continue  # an optional attribute left unset is omitted
        attrs[field.h5_name] = _serialize_attribute(field, value, apath)
    return attrs


def _add_snapshot_attrs(
    foreign: ForeignSpec,
    record: Any,
    path: str,
    attrs: dict[str, Any],
    declared: set[str],
) -> None:
    """Copy the undeclared entries of the ``attrs=`` snapshot field, raw."""
    if foreign.attrs is None:
        return
    snapshot = _read_field(foreign, record, foreign.attrs)
    if snapshot is None:
        return
    if not isinstance(snapshot, Mapping):
        raise ValidationError(path, f"attrs field {foreign.attrs!r} must be a Mapping")
    for name, value in snapshot.items():
        name = str(name)
        if name in declared:
            continue  # declared names always come from their typed fields
        apath = attr_path(path, name)
        if foreign.spec.extras is Extras.FORBID:
            raise ValidationError(apath, "unexpected attribute")
        _check_storable(value, apath)
        attrs[name] = value


def _materialize(lazy: LazyArray, path: str) -> np.ndarray:
    """Return a lazy payload's data without caching it on the caller's record."""
    cached = lazy._h5t_data
    if cached is not _DATA_NOT_LOADED:
        return typing.cast(np.ndarray, cached)
    try:
        with lazy.open() as dataset:
            return np.asarray(dataset[()])
    except OSError as exc:
        raise ValidationError(path, f"cannot read lazy payload: {exc}") from exc


def _plan_payload(value: Any, array: ArraySpec | None, path: str, *, lazy: bool) -> np.ndarray:
    """Check a payload (``np.ndarray`` or ``LazyArray``) and return its array."""
    expected = LazyArray if lazy else np.ndarray
    if not isinstance(value, expected):
        raise ValidationError(path, f"expected {expected.__name__}, got {type(value).__name__}")
    if array is not None:
        _check_array(array, value, path)
    if isinstance(value, LazyArray):
        return _materialize(value, path)
    return np.asarray(value)


def _plan_dataset_record(foreign: ForeignSpec, record: Any, path: str) -> _DatasetPlan:
    assert foreign.data is not None
    # A dataset record is its own binding scope, as on load.
    with jaxtyped("context"):
        attrs = _plan_attributes(foreign, record, path, foreign.spec.fields)
        declared = {f.h5_name for f in foreign.spec.fields}
        _add_snapshot_attrs(foreign, record, path, attrs, declared)
        payload = _plan_payload(
            _read_field(foreign, record, foreign.data),
            foreign.payload_array,
            path,
            lazy=foreign.lazy_type is not None,
        )
    return _DatasetPlan(payload, attrs)


def _plan_member(field: FieldSpec, value: Any, path: str) -> _GroupPlan | _DatasetPlan:
    """Plan one validated non-attribute, non-collection member (twin of ``_load_member``)."""
    if field.kind is MemberKind.GROUP:
        assert field.member_type is not None
        return _plan_group(_compiled_record(field.member_type), value, path)
    if field.kind is MemberKind.DATASET:
        if field.foreign is not None:
            return _plan_dataset_record(field.foreign, value, path)
        # A bare LazyArray member: no record type, no declared attributes.
        return _DatasetPlan(_plan_payload(value, field.array, path, lazy=True), {})
    return _DatasetPlan(_plan_payload(value, field.array, path, lazy=False), {})


def _plan_collection(field: FieldSpec, value: Mapping[str, Any], path: str) -> _GroupPlan:
    assert field.item is not None
    plan = _GroupPlan()
    for key, item in value.items():
        item_path = child_path(path, str(key))
        if not isinstance(key, str) or not key or "/" in key or key in (".", ".."):
            raise ValidationError(
                item_path, "collection key must be a non-empty str without '/' and not '.' or '..'"
            )
        if field.glob is not None and field.glob.match(key) is None:
            raise ValidationError(item_path, "collection key does not match the field's Glob")
        validated = _validate_value(field.item, item, item_path)
        plan.children[key] = _plan_member(field.item, validated, item_path)
    return plan


def _plan_group(foreign: ForeignSpec, record: Any, path: str) -> _GroupPlan:
    """Plan ``record`` as a group (twin of ``_load_group_values``)."""
    # Named dimensions unify across this record's own array fields only.
    with jaxtyped("context"):
        plan = _GroupPlan()
        plan.attrs = _plan_attributes(foreign, record, path, foreign.spec.fields)
        for field in foreign.spec.fields:
            if field.kind is MemberKind.ATTRIBUTE:
                continue
            member_path = child_path(path, field.h5_name)
            value = _validate_value(field, _read_field(foreign, record, field.py_name), member_path)
            if value is None:
                continue  # an optional child left unset is omitted
            if field.kind is MemberKind.COLLECTION:
                plan.children[field.h5_name] = _plan_collection(field, value, member_path)
            else:
                plan.children[field.h5_name] = _plan_member(field, value, member_path)
        declared = {f.h5_name for f in foreign.spec.fields if f.kind is MemberKind.ATTRIBUTE}
        _add_snapshot_attrs(foreign, record, path, plan.attrs, declared)
    return plan


# ---------------------------------------------------------------------------
# Phase 2: apply
# ---------------------------------------------------------------------------


def _write_attrs(node: h5py.Group | h5py.Dataset, attrs: Mapping[str, Any], path: str) -> None:
    for name, value in attrs.items():
        try:
            node.attrs[name] = value
        except (TypeError, ValueError) as exc:
            raise ValidationError(attr_path(path, name), f"cannot write attribute: {exc}") from exc


def _write_plan(group: h5py.Group, plan: _GroupPlan, path: str) -> None:
    _write_attrs(group, plan.attrs, path)
    for name, child in plan.children.items():
        cpath = child_path(path, name)
        if isinstance(child, _GroupPlan):
            _write_plan(group.create_group(name), child, cpath)
            continue
        try:
            dataset = group.create_dataset(name, data=child.data)
        except (TypeError, ValueError) as exc:
            raise ValidationError(cpath, f"cannot write dataset: {exc}") from exc
        _write_attrs(dataset, child.attrs, cpath)


def _clear(node: h5py.Group) -> None:
    for name in list(node.keys()):
        del node[name]
    for name in list(node.attrs.keys()):
        del node.attrs[name]


@dataclasses.dataclass(frozen=True)
class _Undo:
    """How to take back what ``_write_plan`` adds: delete a created group, or clear one."""

    delete: str | None = None
    clear: str | None = None

    def rollback(self, file: h5py.File) -> None:
        with contextlib.suppress(Exception):  # best effort: never mask the original error
            if self.delete is not None:
                del file[self.delete]
            elif self.clear is not None:
                _clear(typing.cast(h5py.Group, file[self.clear]))


def _first_missing(file: h5py.File, root: str) -> str:
    """Return the shallowest ancestor of ``root`` (or ``root``) that does not exist yet."""
    current = ""
    for part in root.strip("/").split("/"):
        current = f"{current}/{part}"
        if current not in file:
            return current
    return root  # unreachable: ``root`` was just found missing


def _prepare_target(file: h5py.File, root: str, overwrite: bool) -> _Undo:
    try:
        node = file.get(root)
    except (TypeError, ValueError, KeyError) as exc:
        raise ValidationError(root, f"cannot resolve root: {exc}") from exc
    if node is not None:
        if not isinstance(node, h5py.Group):
            raise ValidationError(root, "expected a group, found a dataset")
        if len(node) or len(node.attrs):
            if not overwrite:
                raise ValidationError(root, "group already exists and is not empty")
            _clear(node)
        return _Undo(clear=root)
    first_missing = _first_missing(file, root)
    try:
        file.require_group(root)
    except (TypeError, ValueError) as exc:
        raise ValidationError(root, f"cannot create root: {exc}") from exc
    return _Undo(delete=first_missing)


def _apply(
    filename: str,
    mode: Literal["w", "w-", "a"],
    root: str,
    plan: _GroupPlan,
    overwrite: bool,
) -> None:
    created_file = mode != "a" or not os.path.exists(filename)
    h5file = h5py.File(filename, mode)  # a failure to open must not remove anything
    ok = False
    undo = _Undo()
    try:
        undo = _prepare_target(h5file, root, overwrite)
        _write_plan(typing.cast(h5py.Group, h5file[root]), plan, root)
        ok = True
    finally:
        if not ok and not created_file:
            undo.rollback(h5file)
        h5file.close()
        if not ok and created_file:
            with contextlib.suppress(OSError):
                os.remove(filename)


def save(
    record: object,
    path: str | os.PathLike[str],
    root: str = "/",
    mode: Literal["w-", "x", "w", "a"] = "w-",
    overwrite: bool = False,
) -> None:
    """Validate ``record`` against its schema and write it under ``root`` in ``path``.

    Everything is validated and serialized in memory before the file is opened, so a bad
    record never touches the target. If writing itself fails, what was written is
    removed: the new file, or (for ``mode="a"``, best effort) the group save created.

    Parameters
    ----------
    record : object
        An instance of a ``@h5t.group``-decorated record type.
    path : str or os.PathLike
        Target HDF5 file.
    root : str
        Absolute group path to write under; missing groups are created.
    mode : {"w-", "x", "w", "a"}
        ``"w-"``/``"x"`` fail if the file exists, ``"w"`` truncates, ``"a"`` writes into
        an existing file or creates one.
    overwrite : bool
        Allow replacing the contents of a non-empty ``root`` group. This is destructive
        and not undone if the write then fails.

    Raises
    ------
    SchemaError
        ``record`` is not an instance of a ``@h5t.group`` record.
    ValidationError
        The record does not conform to its schema, or ``root`` is occupied.
    ConversionError
        An ``Attr(serializer=...)`` raised.
    ValueError
        ``mode`` or ``root`` is invalid.
    """
    if mode not in ("w-", "x", "w", "a"):
        raise ValueError(f"mode must be 'w-', 'x', 'w' or 'a', got {mode!r}")
    schema = type(record)
    if "__h5t_record__" not in schema.__dict__:
        raise SchemaError(f"{schema!r} is not a decorated record (@h5t.dataset/@h5t.group)")
    foreign = _compiled_record(schema)
    if foreign.kind is not RecordKind.GROUP:
        raise SchemaError(
            f"{schema.__name__} is a dataset record (@h5t.dataset); h5t.save needs a "
            "@h5t.group record"
        )
    normalized_root = _normalize_root(root)
    filename = _filesystem_path(path)

    plan = _plan_group(foreign, record, normalized_root)
    _apply(filename, "w-" if mode == "x" else mode, normalized_root, plan, overwrite)
