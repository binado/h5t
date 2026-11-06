"""Schema-validated writing of `Group` instances to HDF5, in two phases.

Pydantic validated the `Group` when it was constructed (field types, jaxtyping shapes and
dtypes, attribute models), so the writer does not check any of that again. Phase 1 (*plan*)
walks the compiled spec over the instance and builds a tree of what to write, checking only
what HDF5 itself imposes (child names, storable attribute values) and reading the data of
every lazy `Dataset`. It never touches the target file. Phase 2 (*apply*) opens the file and
writes the tree, rolling back what it wrote if anything fails.
"""

from __future__ import annotations

import dataclasses
import os
import typing
from collections.abc import Mapping
from enum import Enum
from typing import Any, Literal

import h5py
import numpy as np
from pydantic_core import PydanticSerializationError

from h5t._array import _DATA_NOT_LOADED, Dataset
from h5t._compile import Group, _compiled_group, _filesystem_path, _normalize_root
from h5t._errors import SchemaError, ValidationError
from h5t._spec import AttrsSpec, FieldSpec, GroupSpec, MemberKind, attr_path, child_path


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


def _read_field(owner_type: type, instance: Any, name: str) -> Any:
    try:
        return getattr(instance, name)
    except AttributeError as exc:
        raise SchemaError(f"{owner_type.__name__}.{name}: model has no such attribute") from exc


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
            f"cannot store {type(value).__name__} as an HDF5 attribute; add a PlainSerializer",
        )


def _attr_values(attrs_spec: AttrsSpec | None, attrs_val: Any, path: str) -> Mapping[str, Any]:
    """Return the attribute names and values to store, by their on-disk names."""
    if attrs_spec is None:
        return typing.cast(Mapping[str, Any], attrs_val)
    # ``include`` limits the dump to declared fields, so computed fields are not stored.
    declared = set(type(attrs_val).model_fields)
    try:
        return attrs_val.model_dump(mode="python", by_alias=True, include=declared)
    except (PydanticSerializationError, TypeError) as exc:
        cause = (
            exc.__cause__ if isinstance(exc, PydanticSerializationError) and exc.__cause__ else exc
        )
        raise ValidationError(path, f"attribute serializer failed: {cause}") from cause


def _plan_attrs(
    attrs_spec: AttrsSpec | None,
    attrs_val: Any,
    path: str,
    *,
    forbid_untyped_extras: bool = False,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for raw_name, value in _attr_values(attrs_spec, attrs_val, path).items():
        name = str(raw_name)
        if value is None and attrs_spec is not None and name in attrs_spec.optional:
            continue
        apath = attr_path(path, name)
        if forbid_untyped_extras:
            raise ValidationError(apath, "unexpected attribute")
        if isinstance(value, Enum):
            value = value.value
        _check_storable(value, apath)
        out[name] = value
    return out


def _materialize(dataset: Dataset[Any], path: str) -> np.ndarray:
    """Return a dataset's array data without mutating the caller's detached instance."""
    cached = dataset._h5t_data
    if cached is not _DATA_NOT_LOADED:
        return typing.cast(np.ndarray, cached)
    try:
        with dataset.open() as h5ds:
            return np.asarray(h5ds[()])
    except (ValidationError, OSError) as exc:
        detail = exc.message if isinstance(exc, ValidationError) else str(exc)
        raise ValidationError(path, f"cannot read dataset data: {detail}") from exc


def _plan_dataset(value: Dataset[Any], attrs_spec: AttrsSpec | None, path: str) -> _DatasetPlan:
    attrs = _plan_attrs(attrs_spec, value.attrs, path)
    return _DatasetPlan(data=_materialize(value, path), attrs=attrs)


def _is_omitted(field: FieldSpec, value: Any) -> bool:
    """Report whether ``value`` is an optional member left as ``None``."""
    return value is None and field.optional


def _plan_member(field: FieldSpec, value: Any, path: str) -> _GroupPlan | _DatasetPlan:
    if field.kind is MemberKind.GROUP:
        assert field.member_type is not None
        group_cls = typing.cast("type[Group[Any]]", field.member_type)
        return _plan_group(_compiled_group(group_cls), value, path)
    if field.kind is MemberKind.DATASET:
        return _plan_dataset(value, field.dataset_attrs, path)
    return _DatasetPlan(data=np.asarray(value), attrs={})


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
        if not _is_omitted(field.item, item):
            plan.children[key] = _plan_member(field.item, item, item_path)
    return plan


def _plan_group(spec: GroupSpec, group: Any, path: str) -> _GroupPlan:
    plan = _GroupPlan()
    plan.attrs = _plan_attrs(
        spec.attrs,
        _read_field(spec.group_type, group, "attrs"),
        path,
        forbid_untyped_extras=(spec.forbid_extras and spec.attrs is None),
    )
    for field in spec.fields:
        member_path = child_path(path, field.h5_name)
        raw_val = _read_field(spec.group_type, group, field.py_name)
        if _is_omitted(field, raw_val):
            continue
        if field.kind is MemberKind.COLLECTION:
            plan.children[field.h5_name] = _plan_collection(field, raw_val, member_path)
        else:
            plan.children[field.h5_name] = _plan_member(field, raw_val, member_path)
    return plan


# ---------------------------------------------------------------------------
# Phase 2: apply
# ---------------------------------------------------------------------------


def _write_attrs(node: h5py.Group | h5py.Dataset, attrs: Mapping[str, Any], path: str) -> None:
    for name, value in attrs.items():
        try:
            node.attrs[name] = value
        except (TypeError, ValueError, OSError) as exc:
            raise ValidationError(attr_path(path, name), f"cannot write attribute: {exc}") from exc


def _write_plan(group: h5py.Group, plan: _GroupPlan, path: str) -> None:
    _write_attrs(group, plan.attrs, path)
    for name, child in plan.children.items():
        cpath = child_path(path, name)
        kind = "group" if isinstance(child, _GroupPlan) else "dataset"
        try:
            if isinstance(child, _GroupPlan):
                node = group.create_group(name)
            else:
                node = group.create_dataset(name, data=child.data)
        except (TypeError, ValueError, OSError) as exc:
            raise ValidationError(cpath, f"cannot write {kind}: {exc}") from exc
        if isinstance(child, _GroupPlan):
            _write_plan(node, child, cpath)
        else:
            _write_attrs(node, child.attrs, cpath)


def _clear(node: h5py.Group) -> None:
    node.clear()
    node.attrs.clear()


@dataclasses.dataclass
class _Undo:
    """What to remove if writing fails; filled in *before* the file is mutated."""

    delete: str | None = None
    clear: str | None = None

    def rollback(self, file: h5py.File) -> None:
        if self.delete is not None:
            if self.delete in file:
                del file[self.delete]
        elif self.clear is not None:
            _clear(typing.cast(h5py.Group, file[self.clear]))


def _first_missing(file: h5py.File, root: str) -> str:
    current = ""
    for part in root.strip("/").split("/"):
        current = f"{current}/{part}"
        if current not in file:
            return current
    return root


def _prepare_target(file: h5py.File, root: str, overwrite: bool, undo: _Undo) -> None:
    """Make ``root`` writable, clearing it first when ``overwrite`` is set.

    ``undo`` is updated before each mutation so a failure midway can still be rolled
    back. Like ``del group[name]`` in h5py, clearing existing content is destructive and
    is not undone if the write then fails; rollback only empties the group again.
    """
    try:
        node = file.get(root)
    except (TypeError, ValueError, KeyError) as exc:
        raise ValidationError(root, f"cannot resolve root: {exc}") from exc
    if node is not None:
        if not isinstance(node, h5py.Group):
            raise ValidationError(root, "expected a group, found a dataset")
        if (len(node) or len(node.attrs)) and not overwrite:
            raise ValidationError(root, "group already exists and is not empty")
        undo.clear = root
        _clear(node)
        return
    undo.delete = _first_missing(file, root)
    try:
        file.require_group(root)
    except (TypeError, ValueError, OSError) as exc:
        raise ValidationError(root, f"cannot create root: {exc}") from exc


def _apply(
    filename: str,
    mode: Literal["w", "w-", "a"],
    root: str,
    plan: _GroupPlan,
    overwrite: bool,
) -> None:
    created_file = mode != "a" or not os.path.exists(filename)
    undo = _Undo()
    with h5py.File(filename, mode) as h5file:
        try:
            _prepare_target(h5file, root, overwrite, undo)
            _write_plan(typing.cast(h5py.Group, h5file[root]), plan, root)
        except BaseException as exc:
            _roll_back(h5file, filename, undo, created_file, exc)
            raise


def _roll_back(
    h5file: h5py.File,
    filename: str,
    undo: _Undo,
    created_file: bool,
    exc: BaseException,
) -> None:
    """Undo a failed write, attaching any rollback problem to ``exc`` as a note."""
    if created_file:
        h5file.close()
        try:
            os.remove(filename)
        except OSError as remove_exc:
            exc.add_note(
                f"h5t: could not remove the partially written file {filename!r}: {remove_exc}"
            )
        return
    try:
        undo.rollback(h5file)
    except Exception as rollback_exc:
        exc.add_note(
            f"h5t: rollback failed, {filename!r} may contain partially written data: {rollback_exc}"
        )


def _dump_group(
    group: Group[Any],
    path: str | os.PathLike[str],
    root: str = "/",
    mode: Literal["w-", "x", "w", "a"] = "w-",
    overwrite: bool = False,
) -> None:
    if mode not in ("w-", "x", "w", "a"):
        raise ValueError(f"mode must be 'w-', 'x', 'w' or 'a', got {mode!r}")
    spec = _compiled_group(type(group))
    normalized_root = _normalize_root(root)
    filename = _filesystem_path(path)

    plan = _plan_group(spec, group, normalized_root)
    _apply(filename, "w-" if mode == "x" else mode, normalized_root, plan, overwrite)
